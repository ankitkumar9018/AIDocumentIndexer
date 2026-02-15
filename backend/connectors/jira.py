"""
Jira Connector for AIDocumentIndexer
====================================

Fetches issues, comments, and attachments from Atlassian Jira.
Supports:
- Cloud and Data Center/Server instances
- Project filtering
- JQL queries
- Issue comments and attachments
- Incremental sync
"""

import asyncio
import base64
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JiraIssue(BaseModel):
    """Represents a Jira issue."""
    id: str
    key: str  # e.g., PROJ-123
    summary: str
    description: str = ""
    issue_type: str = ""
    status: str = ""
    priority: str = ""
    project_key: str = ""
    project_name: str = ""
    reporter: str = ""
    assignee: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    labels: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    fix_versions: list[str] = Field(default_factory=list)
    comments: list[dict] = Field(default_factory=list)
    attachments: list[dict] = Field(default_factory=list)
    custom_fields: dict = Field(default_factory=dict)
    url: str = ""
    parent_key: Optional[str] = None  # For subtasks
    subtasks: list[str] = Field(default_factory=list)
    linked_issues: list[dict] = Field(default_factory=list)

    @property
    def content(self) -> str:
        """Get searchable content."""
        parts = [self.summary, self.description]
        for comment in self.comments:
            parts.append(comment.get("body", ""))
        return "\n\n".join(filter(None, parts))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.key}:{self.summary}:{self.updated_at}".encode()).hexdigest()[:16]


class JiraComment(BaseModel):
    """Represents a Jira comment."""
    id: str
    issue_key: str
    author: str
    body: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class JiraAttachment(BaseModel):
    """Represents a Jira attachment."""
    id: str
    issue_key: str
    filename: str
    mime_type: str
    size: int
    author: str = ""
    created_at: Optional[datetime] = None
    download_url: str = ""


class JiraProject(BaseModel):
    """Represents a Jira project."""
    id: str
    key: str
    name: str
    description: str = ""
    lead: str = ""
    url: str = ""
    issue_types: list[str] = Field(default_factory=list)


class JiraConnectorConfig(BaseModel):
    """Configuration for Jira connector."""
    # Authentication
    base_url: str  # e.g., https://yourcompany.atlassian.net
    username: str  # email for Cloud
    api_token: str  # API token for Cloud, password for Server/DC

    # Instance type
    is_cloud: bool = True  # Cloud vs Data Center/Server

    # Sync settings
    project_keys: list[str] = Field(default_factory=list)  # Empty = all projects
    issue_types: list[str] = Field(default_factory=list)  # Empty = all types
    statuses: list[str] = Field(default_factory=list)  # Empty = all statuses
    include_attachments: bool = True
    include_comments: bool = True
    max_attachment_size_mb: int = 50
    max_issues: int = 10000
    sync_interval_minutes: int = 30

    # JQL filters
    jql_filter: Optional[str] = None  # Additional JQL conditions


class JiraConnector:
    """
    Jira API connector for fetching issues and project data.

    Usage:
        connector = JiraConnector(config)
        async for issue in connector.fetch_issues():
            print(f"Issue: {issue.key} - {issue.summary}")
    """

    @staticmethod
    def _validate_jql(jql: str) -> str:
        """Validate JQL to prevent injection."""
        dangerous = re.compile(r'\b(ORDER\s+BY|LIMIT|OFFSET)\b', re.IGNORECASE)
        if dangerous.search(jql):
            raise ValueError("JQL cannot contain ORDER BY, LIMIT, or OFFSET")
        if jql.count('"') % 2 != 0:
            raise ValueError("JQL has unbalanced quotes")
        if len(jql) > 1000:
            raise ValueError("JQL too long")
        return jql

    def __init__(self, config: JiraConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

        # API version
        if config.is_cloud:
            self.api_base = f"{config.base_url.rstrip('/')}/rest/api/3"
        else:
            self.api_base = f"{config.base_url.rstrip('/')}/rest/api/2"

    @property
    def _auth_header(self) -> str:
        """Generate Basic Auth header."""
        credentials = f"{self.config.username}:{self.config.api_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    @property
    def headers(self) -> dict:
        return {
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=60.0,
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
        url = f"{self.api_base}{endpoint}"
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Jira API error {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Jira request error: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test the API connection."""
        data = await self._request("GET", "/myself")
        return data is not None

    async def get_current_user(self) -> Optional[dict]:
        """Get current user info."""
        return await self._request("GET", "/myself")

    async def list_projects(self) -> AsyncGenerator[JiraProject, None]:
        """List all accessible projects."""
        start = 0
        max_results = 50

        while True:
            params = {
                "startAt": start,
                "maxResults": max_results,
                "expand": "description,lead,issueTypes",
            }

            data = await self._request("GET", "/project/search", params=params)
            if not data:
                # Try alternative endpoint for Server/DC
                data = await self._request("GET", "/project", params=params)
                if not data:
                    break

            projects = data.get("values", data) if isinstance(data, dict) else data

            for proj_data in projects:
                # Filter by configured project keys
                if self.config.project_keys and proj_data["key"] not in self.config.project_keys:
                    continue

                issue_types = []
                for it in proj_data.get("issueTypes", []):
                    issue_types.append(it.get("name", ""))

                lead_name = ""
                lead = proj_data.get("lead", {})
                if lead:
                    lead_name = lead.get("displayName", lead.get("name", ""))

                yield JiraProject(
                    id=proj_data["id"],
                    key=proj_data["key"],
                    name=proj_data["name"],
                    description=proj_data.get("description", ""),
                    lead=lead_name,
                    url=f"{self.config.base_url}/browse/{proj_data['key']}",
                    issue_types=issue_types,
                )

            # Check if more results
            if isinstance(data, dict):
                total = data.get("total", 0)
                if start + max_results >= total:
                    break
            elif len(projects) < max_results:
                break

            start += max_results
            await asyncio.sleep(0.1)

    async def fetch_issues(
        self,
        project_key: Optional[str] = None,
        jql: Optional[str] = None,
        modified_since: Optional[datetime] = None,
        max_issues: Optional[int] = None,
    ) -> AsyncGenerator[JiraIssue, None]:
        """
        Fetch issues from Jira using JQL.

        Args:
            project_key: Specific project to fetch from
            jql: Custom JQL query (overrides other filters)
            modified_since: Only fetch issues modified after this date
            max_issues: Maximum issues to fetch

        Yields:
            JiraIssue objects
        """
        max_issues = max_issues or self.config.max_issues
        count = 0

        # Build JQL query
        if jql:
            full_jql = self._validate_jql(jql)
        else:
            jql_parts = []

            if project_key:
                if not re.match(r'^[A-Za-z][A-Za-z0-9_-]{0,49}$', project_key):
                    raise ValueError(f"Invalid project key format: {project_key}")
                jql_parts.append(f"project = {project_key}")
            elif self.config.project_keys:
                for pk in self.config.project_keys:
                    if not re.match(r'^[A-Za-z][A-Za-z0-9_-]{0,49}$', pk):
                        raise ValueError(f"Invalid project key format: {pk}")
                projects = ", ".join(self.config.project_keys)
                jql_parts.append(f"project IN ({projects})")

            if self.config.issue_types:
                for t in self.config.issue_types:
                    if not re.match(r'^[A-Za-z][A-Za-z0-9 _-]{0,49}$', t):
                        raise ValueError(f"Invalid issue type format: {t}")
                types = ", ".join(f'"{t}"' for t in self.config.issue_types)
                jql_parts.append(f"issuetype IN ({types})")

            if self.config.statuses:
                for s in self.config.statuses:
                    if not re.match(r'^[A-Za-z][A-Za-z0-9 _-]{0,49}$', s):
                        raise ValueError(f"Invalid status format: {s}")
                statuses = ", ".join(f'"{s}"' for s in self.config.statuses)
                jql_parts.append(f"status IN ({statuses})")

            if modified_since:
                date_str = modified_since.strftime("%Y-%m-%d %H:%M")
                jql_parts.append(f'updated >= "{date_str}"')

            if self.config.jql_filter:
                jql_parts.append(f"({self._validate_jql(self.config.jql_filter)})")

            full_jql = " AND ".join(jql_parts) if jql_parts else ""
            full_jql += " ORDER BY updated DESC"

        start = 0
        max_results = 50

        # Fields to fetch
        fields = [
            "summary", "description", "issuetype", "status", "priority",
            "project", "reporter", "assignee", "created", "updated",
            "resolutiondate", "labels", "components", "fixVersions",
            "parent", "subtasks", "issuelinks",
        ]

        if self.config.include_comments:
            fields.append("comment")
        if self.config.include_attachments:
            fields.append("attachment")

        while count < max_issues:
            params = {
                "jql": full_jql.strip(),
                "startAt": start,
                "maxResults": min(max_results, max_issues - count),
                "fields": ",".join(fields),
            }

            data = await self._request("GET", "/search", params=params)
            if not data:
                break

            for issue_data in data.get("issues", []):
                issue = self._parse_issue(issue_data)
                if issue:
                    count += 1
                    yield issue

                    if count >= max_issues:
                        break

            # Check for more results
            total = data.get("total", 0)
            if start + max_results >= total:
                break
            start += max_results
            await asyncio.sleep(0.1)

    def _parse_issue(self, data: dict) -> Optional[JiraIssue]:
        """Parse issue data from API response."""
        try:
            fields = data.get("fields", {})

            # Parse dates
            created_at = None
            updated_at = None
            resolved_at = None

            if fields.get("created"):
                created_at = datetime.fromisoformat(fields["created"].replace("Z", "+00:00"))
            if fields.get("updated"):
                updated_at = datetime.fromisoformat(fields["updated"].replace("Z", "+00:00"))
            if fields.get("resolutiondate"):
                resolved_at = datetime.fromisoformat(fields["resolutiondate"].replace("Z", "+00:00"))

            # Parse user fields
            reporter = ""
            if fields.get("reporter"):
                reporter = fields["reporter"].get("displayName", fields["reporter"].get("name", ""))

            assignee = ""
            if fields.get("assignee"):
                assignee = fields["assignee"].get("displayName", fields["assignee"].get("name", ""))

            # Parse project
            project = fields.get("project", {})

            # Parse type, status, priority
            issue_type = fields.get("issuetype", {}).get("name", "")
            status = fields.get("status", {}).get("name", "")
            priority = fields.get("priority", {}).get("name", "") if fields.get("priority") else ""

            # Parse labels
            labels = fields.get("labels", [])

            # Parse components
            components = [c.get("name", "") for c in fields.get("components", [])]

            # Parse fix versions
            fix_versions = [v.get("name", "") for v in fields.get("fixVersions", [])]

            # Parse comments
            comments = []
            comment_data = fields.get("comment", {})
            if comment_data:
                for comment in comment_data.get("comments", []):
                    author = comment.get("author", {})
                    comments.append({
                        "id": comment.get("id"),
                        "author": author.get("displayName", author.get("name", "")),
                        "body": self._extract_text_from_adf(comment.get("body", "")),
                        "created": comment.get("created"),
                    })

            # Parse attachments
            attachments = []
            for att in fields.get("attachment", []):
                file_size = att.get("size", 0)
                max_size = self.config.max_attachment_size_mb * 1024 * 1024
                if file_size <= max_size:
                    author = att.get("author", {})
                    attachments.append({
                        "id": att.get("id"),
                        "filename": att.get("filename", ""),
                        "mime_type": att.get("mimeType", ""),
                        "size": file_size,
                        "author": author.get("displayName", author.get("name", "")),
                        "url": att.get("content", ""),
                    })

            # Parse parent (for subtasks)
            parent_key = None
            if fields.get("parent"):
                parent_key = fields["parent"].get("key")

            # Parse subtasks
            subtasks = [st.get("key", "") for st in fields.get("subtasks", [])]

            # Parse linked issues
            linked_issues = []
            for link in fields.get("issuelinks", []):
                link_type = link.get("type", {}).get("name", "")
                if link.get("inwardIssue"):
                    linked_issues.append({
                        "key": link["inwardIssue"].get("key"),
                        "type": link_type,
                        "direction": "inward",
                    })
                if link.get("outwardIssue"):
                    linked_issues.append({
                        "key": link["outwardIssue"].get("key"),
                        "type": link_type,
                        "direction": "outward",
                    })

            # Parse description
            description = self._extract_text_from_adf(fields.get("description", ""))

            return JiraIssue(
                id=data["id"],
                key=data["key"],
                summary=fields.get("summary", ""),
                description=description,
                issue_type=issue_type,
                status=status,
                priority=priority,
                project_key=project.get("key", ""),
                project_name=project.get("name", ""),
                reporter=reporter,
                assignee=assignee,
                created_at=created_at,
                updated_at=updated_at,
                resolved_at=resolved_at,
                labels=labels,
                components=components,
                fix_versions=fix_versions,
                comments=comments,
                attachments=attachments,
                url=f"{self.config.base_url}/browse/{data['key']}",
                parent_key=parent_key,
                subtasks=subtasks,
                linked_issues=linked_issues,
            )

        except Exception as e:
            logger.error(f"Error parsing issue {data.get('key')}: {e}")
            return None

    def _extract_text_from_adf(self, content: Any) -> str:
        """
        Extract text from Atlassian Document Format (ADF) or plain text.
        ADF is used in Jira Cloud API v3.
        """
        if not content:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            # ADF format
            text_parts = []

            def extract_text(node: dict):
                if node.get("type") == "text":
                    text_parts.append(node.get("text", ""))
                for child in node.get("content", []):
                    if isinstance(child, dict):
                        extract_text(child)

            extract_text(content)
            return " ".join(text_parts)

        return str(content)

    async def get_issue(self, issue_key: str) -> Optional[JiraIssue]:
        """Get a specific issue by key."""
        fields = [
            "summary", "description", "issuetype", "status", "priority",
            "project", "reporter", "assignee", "created", "updated",
            "resolutiondate", "labels", "components", "fixVersions",
            "parent", "subtasks", "issuelinks", "comment", "attachment",
        ]

        data = await self._request(
            "GET",
            f"/issue/{issue_key}",
            params={"fields": ",".join(fields)},
        )

        if not data:
            return None

        return self._parse_issue(data)

    async def get_issue_comments(self, issue_key: str) -> AsyncGenerator[JiraComment, None]:
        """Get all comments for an issue."""
        start = 0
        max_results = 100

        while True:
            data = await self._request(
                "GET",
                f"/issue/{issue_key}/comment",
                params={"startAt": start, "maxResults": max_results},
            )

            if not data:
                break

            for comment in data.get("comments", []):
                author = comment.get("author", {})
                created_at = None
                updated_at = None

                if comment.get("created"):
                    created_at = datetime.fromisoformat(comment["created"].replace("Z", "+00:00"))
                if comment.get("updated"):
                    updated_at = datetime.fromisoformat(comment["updated"].replace("Z", "+00:00"))

                yield JiraComment(
                    id=comment["id"],
                    issue_key=issue_key,
                    author=author.get("displayName", author.get("name", "")),
                    body=self._extract_text_from_adf(comment.get("body", "")),
                    created_at=created_at,
                    updated_at=updated_at,
                )

            total = data.get("total", 0)
            if start + max_results >= total:
                break
            start += max_results

    async def download_attachment(self, attachment_url: str) -> Optional[bytes]:
        """Download attachment content."""
        client = await self._get_client()
        try:
            response = await client.get(attachment_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading attachment: {e}")
            return None

    async def search_issues(
        self,
        query: str,
        project_key: Optional[str] = None,
        limit: int = 20,
    ) -> AsyncGenerator[JiraIssue, None]:
        """
        Search issues using text search.

        Args:
            query: Search text
            project_key: Limit to specific project
            limit: Maximum results

        Yields:
            Matching JiraIssue objects
        """
        safe_query = query.replace('\\', '\\\\').replace('"', '\\"')
        jql = f'text ~ "{safe_query}"'

        if project_key:
            if not re.match(r'^[A-Za-z][A-Za-z0-9_-]{0,49}$', project_key):
                raise ValueError(f"Invalid project key format: {project_key}")
            jql += f" AND project = {project_key}"
        elif self.config.project_keys:
            projects = ", ".join(self.config.project_keys)
            jql += f" AND project IN ({projects})"

        # Note: ORDER BY removed — _validate_jql() rejects it, and Jira
        # returns text-search results sorted by relevance by default.

        count = 0
        async for issue in self.fetch_issues(jql=jql, max_issues=limit):
            count += 1
            yield issue
            if count >= limit:
                break


def create_jira_connector(
    base_url: str,
    username: str,
    api_token: str,
    project_keys: list[str] = None,
    is_cloud: bool = True,
) -> JiraConnector:
    """
    Create a Jira connector instance.

    Args:
        base_url: Jira base URL (e.g., https://company.atlassian.net)
        username: Username or email
        api_token: API token (Cloud) or password (Server/DC)
        project_keys: List of project keys to sync (empty = all)
        is_cloud: True for Atlassian Cloud, False for Server/Data Center

    Returns:
        Configured JiraConnector instance
    """
    config = JiraConnectorConfig(
        base_url=base_url,
        username=username,
        api_token=api_token,
        project_keys=project_keys or [],
        is_cloud=is_cloud,
    )
    return JiraConnector(config)
