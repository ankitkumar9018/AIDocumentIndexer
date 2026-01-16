"""
API client for communicating with Mandala Document Indexer server.
"""

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from .config import get_server_url, get_token


class APIError(Exception):
    """API error with status code and message."""

    def __init__(self, status_code: int, message: str, detail: Optional[str] = None):
        self.status_code = status_code
        self.message = message
        self.detail = detail
        super().__init__(f"{status_code}: {message}")


class APIClient:
    """Client for Mandala Document Indexer API."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.server_url = server_url or get_server_url()
        self.token = token or get_token()

        if not self.server_url:
            raise ValueError("Server URL not configured. Run: mandala-sync login")

        self._client = httpx.Client(
            base_url=self.server_url,
            timeout=60.0,
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _handle_response(self, response: httpx.Response) -> Any:
        """Handle API response and raise errors if needed."""
        if response.status_code == 401:
            raise APIError(401, "Unauthorized", "Please run: mandala-sync login")

        if response.status_code >= 400:
            try:
                error_data = response.json()
                detail = error_data.get("detail", str(error_data))
            except Exception:
                detail = response.text
            raise APIError(response.status_code, f"API Error", detail)

        if response.status_code == 204:
            return None

        return response.json()

    def login(self, email: str, password: str) -> Dict[str, Any]:
        """Login and get authentication token."""
        response = self._client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        return self._handle_response(response)

    def verify_token(self) -> Dict[str, Any]:
        """Verify the current token is valid."""
        response = self._client.get(
            "/api/v1/auth/me",
            headers=self._get_headers(),
        )
        return self._handle_response(response)

    def get_collections(self) -> List[Dict[str, Any]]:
        """Get list of available collections."""
        response = self._client.get(
            "/api/v1/documents/collections/list",
            headers=self._get_headers(),
        )
        data = self._handle_response(response)
        return data.get("collections", [])

    def get_folders(self) -> List[Dict[str, Any]]:
        """Get list of available folders."""
        response = self._client.get(
            "/api/v1/folders",
            headers=self._get_headers(),
        )
        data = self._handle_response(response)
        return data.get("folders", [])

    def get_access_tiers(self) -> List[Dict[str, Any]]:
        """Get list of access tiers."""
        response = self._client.get(
            "/api/v1/admin/access-tiers",
            headers=self._get_headers(),
        )
        return self._handle_response(response)

    def upload_file(
        self,
        file_path: str,
        collection: Optional[str] = None,
        access_tier: int = 1,
        folder_id: Optional[str] = None,
        enable_ocr: bool = True,
    ) -> Dict[str, Any]:
        """Upload a file to the server."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Guess content type
        content_type, _ = mimetypes.guess_type(str(path))
        if not content_type:
            content_type = "application/octet-stream"

        # Prepare form data
        files = {
            "file": (path.name, open(path, "rb"), content_type),
        }

        data = {
            "access_tier": str(access_tier),
            "enable_ocr": "true" if enable_ocr else "false",
        }

        if collection:
            data["collection"] = collection

        if folder_id:
            data["folder_id"] = folder_id

        try:
            response = self._client.post(
                "/api/v1/upload/file",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
                timeout=300.0,  # 5 min timeout for large files
            )
            return self._handle_response(response)
        finally:
            files["file"][1].close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "APIClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
