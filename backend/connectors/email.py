"""
Email Connector for AIDocumentIndexer
=====================================

Fetches emails from Gmail and IMAP servers.
Supports:
- Gmail API
- Generic IMAP
- Email parsing (text, HTML, attachments)
- Label/folder filtering
- Incremental sync
"""

import asyncio
import base64
import email
import hashlib
import imaplib
import logging
import re
from datetime import datetime, timezone, timedelta
from email.header import decode_header
from email.utils import parsedate_to_datetime
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmailMessage(BaseModel):
    """Represents an email message."""
    id: str
    thread_id: Optional[str] = None
    subject: str
    from_address: str
    from_name: str = ""
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    date: datetime
    body_text: str = ""
    body_html: str = ""
    labels: list[str] = Field(default_factory=list)
    attachments: list[dict] = Field(default_factory=list)
    snippet: str = ""
    is_read: bool = True
    is_starred: bool = False

    @property
    def content(self) -> str:
        """Get the best text content."""
        return self.body_text or self._html_to_text(self.body_html)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.subject}:{self.body_text}:{self.date}".encode()).hexdigest()[:16]

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Simple HTML to text conversion."""
        if not html:
            return ""
        # Remove scripts and styles
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Decode entities
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        return text


class EmailAttachment(BaseModel):
    """Represents an email attachment."""
    id: str
    message_id: str
    filename: str
    mime_type: str
    size: int
    content: Optional[bytes] = None


class EmailConnectorConfig(BaseModel):
    """Configuration for Email connector."""
    # Gmail API settings
    gmail_access_token: Optional[str] = None
    gmail_refresh_token: Optional[str] = None

    # IMAP settings
    imap_host: Optional[str] = None
    imap_port: int = 993
    imap_username: Optional[str] = None
    imap_password: Optional[str] = None
    imap_use_ssl: bool = True

    # Sync settings
    labels: list[str] = Field(default_factory=lambda: ["INBOX"])
    folders: list[str] = Field(default_factory=lambda: ["INBOX"])
    include_sent: bool = False
    include_attachments: bool = True
    max_attachment_size_mb: int = 25
    max_emails: int = 1000
    days_to_sync: int = 30
    sync_interval_minutes: int = 60


class GmailConnector:
    """
    Gmail API connector for fetching emails.

    Usage:
        connector = GmailConnector(access_token="ya29.xxx")
        async for email in connector.fetch_emails():
            print(f"Subject: {email.subject}")
    """

    BASE_URL = "https://www.googleapis.com/gmail/v1"

    def __init__(self, config: EmailConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.gmail_access_token}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[dict]:
        client = await self._get_client()
        try:
            response = await client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Gmail API error: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test the API connection."""
        data = await self._request("GET", "/users/me/profile")
        return data is not None

    async def get_profile(self) -> Optional[dict]:
        """Get user profile."""
        return await self._request("GET", "/users/me/profile")

    async def list_labels(self) -> list[dict]:
        """List all labels."""
        data = await self._request("GET", "/users/me/labels")
        return data.get("labels", []) if data else []

    async def fetch_emails(
        self,
        label_ids: Optional[list[str]] = None,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> AsyncGenerator[EmailMessage, None]:
        """
        Fetch emails from Gmail.

        Args:
            label_ids: Filter by labels
            query: Gmail search query
            max_results: Maximum emails to fetch

        Yields:
            EmailMessage objects
        """
        label_ids = label_ids or self.config.labels
        max_results = max_results or self.config.max_emails
        page_token = None
        count = 0

        # Build query for date range
        after_date = datetime.now(timezone.utc) - timedelta(days=self.config.days_to_sync)
        date_query = f"after:{after_date.strftime('%Y/%m/%d')}"
        full_query = f"{query} {date_query}" if query else date_query

        while count < max_results:
            params: dict[str, Any] = {
                "maxResults": min(100, max_results - count),
                "q": full_query,
            }
            if label_ids:
                params["labelIds"] = label_ids
            if page_token:
                params["pageToken"] = page_token

            data = await self._request(
                "GET", "/users/me/messages", params=params
            )
            if not data:
                break

            for msg_ref in data.get("messages", []):
                message = await self._get_message(msg_ref["id"])
                if message:
                    count += 1
                    yield message

                    if count >= max_results:
                        break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            await asyncio.sleep(0.1)

    async def _get_message(self, message_id: str) -> Optional[EmailMessage]:
        """Get full message details."""
        data = await self._request(
            "GET",
            f"/users/me/messages/{message_id}",
            params={"format": "full"},
        )
        if not data:
            return None

        return self._parse_message(data)

    def _parse_message(self, data: dict) -> EmailMessage:
        """Parse Gmail API message response."""
        headers = {
            h["name"].lower(): h["value"]
            for h in data.get("payload", {}).get("headers", [])
        }

        # Parse from address
        from_raw = headers.get("from", "")
        from_match = re.match(r'(?:"?([^"]*)"?\s*)?<?([^>]+)>?', from_raw)
        from_name = from_match.group(1) or "" if from_match else ""
        from_address = from_match.group(2) or from_raw if from_match else from_raw

        # Parse date
        date_str = headers.get("date", "")
        try:
            date = parsedate_to_datetime(date_str)
        except Exception:
            date = datetime.now(timezone.utc)

        # Parse to addresses
        to_raw = headers.get("to", "")
        to_addresses = [addr.strip() for addr in to_raw.split(",") if addr.strip()]

        # Parse cc addresses
        cc_raw = headers.get("cc", "")
        cc_addresses = [addr.strip() for addr in cc_raw.split(",") if addr.strip()]

        # Extract body
        body_text, body_html, attachments = self._extract_body(data.get("payload", {}))

        # Get labels
        labels = data.get("labelIds", [])

        return EmailMessage(
            id=data["id"],
            thread_id=data.get("threadId"),
            subject=headers.get("subject", "(No Subject)"),
            from_address=from_address,
            from_name=from_name,
            to_addresses=to_addresses,
            cc_addresses=cc_addresses,
            date=date,
            body_text=body_text,
            body_html=body_html,
            labels=labels,
            attachments=attachments,
            snippet=data.get("snippet", ""),
            is_read="UNREAD" not in labels,
            is_starred="STARRED" in labels,
        )

    def _extract_body(self, payload: dict) -> tuple[str, str, list[dict]]:
        """Extract body text, HTML, and attachments from payload."""
        body_text = ""
        body_html = ""
        attachments = []

        def process_part(part: dict):
            nonlocal body_text, body_html

            mime_type = part.get("mimeType", "")
            body = part.get("body", {})

            if body.get("attachmentId"):
                # This is an attachment
                attachments.append({
                    "id": body["attachmentId"],
                    "filename": part.get("filename", ""),
                    "mime_type": mime_type,
                    "size": body.get("size", 0),
                })
            elif mime_type == "text/plain" and body.get("data"):
                body_text = base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="ignore")
            elif mime_type == "text/html" and body.get("data"):
                body_html = base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="ignore")
            elif "parts" in part:
                for sub_part in part["parts"]:
                    process_part(sub_part)

        process_part(payload)
        return body_text, body_html, attachments

    async def get_attachment(
        self,
        message_id: str,
        attachment_id: str,
    ) -> Optional[bytes]:
        """Download an attachment."""
        data = await self._request(
            "GET",
            f"/users/me/messages/{message_id}/attachments/{attachment_id}",
        )
        if data and "data" in data:
            return base64.urlsafe_b64decode(data["data"])
        return None


class IMAPConnector:
    """
    Generic IMAP connector for fetching emails.

    Usage:
        connector = IMAPConnector(
            host="imap.example.com",
            username="user@example.com",
            password="password"
        )
        for email in connector.fetch_emails():
            print(f"Subject: {email.subject}")
    """

    def __init__(self, config: EmailConnectorConfig):
        self.config = config
        self._connection: Optional[imaplib.IMAP4_SSL] = None

    def connect(self) -> bool:
        """Connect to IMAP server."""
        try:
            if self.config.imap_use_ssl:
                self._connection = imaplib.IMAP4_SSL(
                    self.config.imap_host,
                    self.config.imap_port,
                )
            else:
                self._connection = imaplib.IMAP4(
                    self.config.imap_host,
                    self.config.imap_port,
                )

            self._connection.login(
                self.config.imap_username,
                self.config.imap_password,
            )
            return True
        except Exception as e:
            logger.error(f"IMAP connection failed: {e}")
            return False

    def close(self):
        """Close IMAP connection."""
        if self._connection:
            try:
                self._connection.logout()
            except Exception:
                pass
            self._connection = None

    def fetch_emails(
        self,
        folder: str = "INBOX",
        since: Optional[datetime] = None,
    ) -> list[EmailMessage]:
        """
        Fetch emails from a folder.

        Args:
            folder: IMAP folder name
            since: Only fetch emails after this date

        Returns:
            List of EmailMessage objects
        """
        if not self._connection:
            if not self.connect():
                return []

        messages = []

        try:
            self._connection.select(folder)

            # Build search criteria
            if since:
                since_str = since.strftime("%d-%b-%Y")
                search_criteria = f'(SINCE {since_str})'
            else:
                search_criteria = "ALL"

            _, message_ids = self._connection.search(None, search_criteria)
            ids = message_ids[0].split()

            # Limit number of messages
            ids = ids[-self.config.max_emails:]

            for msg_id in ids:
                _, msg_data = self._connection.fetch(msg_id, "(RFC822)")
                if msg_data[0] is None:
                    continue

                raw_email = msg_data[0][1]
                message = self._parse_email(raw_email, msg_id.decode())
                if message:
                    messages.append(message)

        except Exception as e:
            logger.error(f"IMAP fetch error: {e}")

        return messages

    def _parse_email(self, raw_email: bytes, msg_id: str) -> Optional[EmailMessage]:
        """Parse raw email bytes into EmailMessage."""
        try:
            msg = email.message_from_bytes(raw_email)

            # Parse subject
            subject = ""
            raw_subject = msg.get("Subject", "")
            if raw_subject:
                decoded = decode_header(raw_subject)
                subject = "".join(
                    part.decode(encoding or "utf-8") if isinstance(part, bytes) else part
                    for part, encoding in decoded
                )

            # Parse from
            from_raw = msg.get("From", "")
            from_match = re.match(r'(?:"?([^"]*)"?\s*)?<?([^>]+)>?', from_raw)
            from_name = from_match.group(1) or "" if from_match else ""
            from_address = from_match.group(2) or from_raw if from_match else from_raw

            # Parse date
            date_str = msg.get("Date", "")
            try:
                date = parsedate_to_datetime(date_str)
            except Exception:
                date = datetime.now(timezone.utc)

            # Parse to
            to_raw = msg.get("To", "")
            to_addresses = [addr.strip() for addr in to_raw.split(",") if addr.strip()]

            # Extract body
            body_text = ""
            body_html = ""
            attachments = []

            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    disposition = str(part.get("Content-Disposition", ""))

                    if "attachment" in disposition:
                        attachments.append({
                            "filename": part.get_filename() or "",
                            "mime_type": content_type,
                            "size": len(part.get_payload(decode=True) or b""),
                        })
                    elif content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_text = payload.decode("utf-8", errors="ignore")
                    elif content_type == "text/html":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_html = payload.decode("utf-8", errors="ignore")
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    if msg.get_content_type() == "text/html":
                        body_html = payload.decode("utf-8", errors="ignore")
                    else:
                        body_text = payload.decode("utf-8", errors="ignore")

            return EmailMessage(
                id=msg_id,
                subject=subject,
                from_address=from_address,
                from_name=from_name,
                to_addresses=to_addresses,
                date=date,
                body_text=body_text,
                body_html=body_html,
                attachments=attachments,
            )

        except Exception as e:
            logger.error(f"Email parse error: {e}")
            return None


def create_gmail_connector(
    access_token: str,
    labels: list[str] = None,
    days_to_sync: int = 30,
) -> GmailConnector:
    """
    Create a Gmail connector instance.

    Args:
        access_token: OAuth access token
        labels: Labels to sync
        days_to_sync: Number of days to look back

    Returns:
        Configured GmailConnector instance
    """
    config = EmailConnectorConfig(
        gmail_access_token=access_token,
        labels=labels or ["INBOX"],
        days_to_sync=days_to_sync,
    )
    return GmailConnector(config)


def create_imap_connector(
    host: str,
    username: str,
    password: str,
    port: int = 993,
    use_ssl: bool = True,
) -> IMAPConnector:
    """
    Create an IMAP connector instance.

    Args:
        host: IMAP server hostname
        username: Email username
        password: Email password
        port: IMAP port
        use_ssl: Use SSL connection

    Returns:
        Configured IMAPConnector instance
    """
    config = EmailConnectorConfig(
        imap_host=host,
        imap_port=port,
        imap_username=username,
        imap_password=password,
        imap_use_ssl=use_ssl,
    )
    return IMAPConnector(config)
