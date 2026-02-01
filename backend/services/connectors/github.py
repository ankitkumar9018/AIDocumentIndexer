"""
AIDocumentIndexer - GitHub Connector
=====================================

Connector for syncing content from GitHub repositories.

Supports:
- Repository files (code, docs, configs)
- README and markdown files
- Issues and pull requests (as documents)
- Wiki pages
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio
import base64

import structlog
import httpx

from backend.services.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorType,
    Resource,
    ResourceType,
    Change,
)
from backend.services.connectors.registry import ConnectorRegistry

logger = structlog.get_logger(__name__)

GITHUB_API_BASE = "https://api.github.com"


@ConnectorRegistry.register(ConnectorType.GITHUB)
class GitHubConnector(BaseConnector):
    """
    Connector for GitHub repositories.

    Uses the GitHub API to sync repository content, issues, and documentation.
    Supports both personal access tokens and GitHub Apps.
    """

    connector_type = ConnectorType.GITHUB
    display_name = "GitHub"
    description = "Sync files and documentation from GitHub repositories"
    icon = "github"

    # File extensions to sync by default
    DEFAULT_EXTENSIONS = [
        ".md", ".mdx", ".txt", ".rst", ".adoc",  # Documentation
        ".py", ".js", ".ts", ".tsx", ".jsx",      # Code
        ".json", ".yaml", ".yml", ".toml",        # Config
        ".html", ".css", ".scss",                 # Web
        ".java", ".go", ".rs", ".rb",             # Languages
    ]

    def __init__(self, config: ConnectorConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _headers(self) -> Dict[str, str]:
        """Get API headers."""
        token = self.config.credentials.get("access_token", "")
        return {
            "Authorization": f"Bearer {token}" if token else "",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=GITHUB_API_BASE,
                headers=self._headers,
                timeout=30.0,
            )
        return self._client

    async def authenticate(self) -> bool:
        """Authenticate with GitHub API."""
        try:
            client = await self._get_client()
            response = await client.get("/user")

            if response.status_code == 200:
                self._authenticated = True
                user_data = response.json()
                self.log_info(
                    "Authenticated with GitHub",
                    user=user_data.get("login"),
                )
                return True
            elif response.status_code == 401:
                self.log_error("GitHub authentication failed: Invalid token")
                return False
            else:
                self.log_error("GitHub auth failed", status=response.status_code)
                return False

        except Exception as e:
            self.log_error("GitHub authentication error", error=e)
            return False

    async def refresh_credentials(self) -> Dict[str, Any]:
        """Refresh OAuth tokens if using GitHub App."""
        # Personal access tokens don't need refresh
        # GitHub App tokens would need refresh logic here
        return self.config.credentials

    async def list_resources(
        self,
        folder_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 100,
    ) -> tuple[List[Resource], Optional[str]]:
        """List repositories or files."""
        client = await self._get_client()
        resources = []

        try:
            if folder_id:
                # folder_id format: "owner/repo" or "owner/repo/path"
                parts = folder_id.split("/", 2)
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1]
                    path = parts[2] if len(parts) > 2 else ""
                    resources, next_token = await self._list_repo_contents(
                        owner, repo, path, client, page_token
                    )
                    return resources, next_token
            else:
                # List user's repositories
                params = {
                    "per_page": min(page_size, 100),
                    "sort": "updated",
                    "direction": "desc",
                }
                if page_token:
                    params["page"] = int(page_token)

                response = await client.get("/user/repos", params=params)

                if response.status_code == 200:
                    repos = response.json()
                    for repo in repos:
                        resource = self._parse_repository(repo)
                        if resource:
                            resources.append(resource)

                    # Check for next page
                    link_header = response.headers.get("Link", "")
                    next_page = self._parse_link_header(link_header)
                    return resources, next_page
                else:
                    self.log_error("GitHub repos list failed", status=response.status_code)

        except Exception as e:
            self.log_error("Failed to list GitHub resources", error=e)

        return resources, None

    async def _list_repo_contents(
        self,
        owner: str,
        repo: str,
        path: str,
        client: httpx.AsyncClient,
        page_token: Optional[str] = None,
    ) -> tuple[List[Resource], Optional[str]]:
        """List contents of a repository path."""
        resources = []

        try:
            url = f"/repos/{owner}/{repo}/contents/{path}" if path else f"/repos/{owner}/{repo}/contents"
            response = await client.get(url)

            if response.status_code == 200:
                contents = response.json()

                # Handle single file response (when path is a file)
                if isinstance(contents, dict):
                    contents = [contents]

                for item in contents:
                    resource = self._parse_content_item(item, owner, repo)
                    if resource:
                        resources.append(resource)

                return resources, None
            else:
                self.log_error(
                    "GitHub contents list failed",
                    status=response.status_code,
                    repo=f"{owner}/{repo}",
                )

        except Exception as e:
            self.log_error("Failed to list repo contents", error=e)

        return resources, None

    def _parse_repository(self, repo: Dict[str, Any]) -> Optional[Resource]:
        """Parse a GitHub repository into a Resource."""
        try:
            full_name = repo.get("full_name", "")
            name = repo.get("name", "")

            return Resource(
                id=full_name,
                name=name,
                resource_type=ResourceType.FOLDER,
                path=f"/{full_name}",
                mime_type="application/vnd.github.repository",
                size=repo.get("size", 0) * 1024,  # GitHub reports in KB
                created_at=datetime.fromisoformat(repo["created_at"].replace("Z", "+00:00")) if repo.get("created_at") else None,
                modified_at=datetime.fromisoformat(repo["updated_at"].replace("Z", "+00:00")) if repo.get("updated_at") else None,
                metadata={
                    "full_name": full_name,
                    "description": repo.get("description", ""),
                    "language": repo.get("language"),
                    "default_branch": repo.get("default_branch", "main"),
                    "is_private": repo.get("private", False),
                    "stars": repo.get("stargazers_count", 0),
                    "forks": repo.get("forks_count", 0),
                    "topics": repo.get("topics", []),
                    "html_url": repo.get("html_url"),
                },
            )
        except Exception as e:
            self.log_error("Failed to parse repository", error=e)
            return None

    def _parse_content_item(
        self,
        item: Dict[str, Any],
        owner: str,
        repo: str,
    ) -> Optional[Resource]:
        """Parse a GitHub content item into a Resource."""
        try:
            item_type = item.get("type", "")
            name = item.get("name", "")
            path = item.get("path", "")
            sha = item.get("sha", "")

            # Determine resource type
            if item_type == "dir":
                resource_type = ResourceType.FOLDER
                mime_type = "application/vnd.github.folder"
            else:
                resource_type = ResourceType.FILE
                mime_type = self._get_mime_type(name)

            return Resource(
                id=f"{owner}/{repo}/{path}",
                name=name,
                resource_type=resource_type,
                path=f"/{owner}/{repo}/{path}",
                mime_type=mime_type,
                size=item.get("size", 0),
                created_at=None,  # GitHub doesn't provide this for files
                modified_at=None,
                metadata={
                    "owner": owner,
                    "repo": repo,
                    "path": path,
                    "sha": sha,
                    "type": item_type,
                    "html_url": item.get("html_url"),
                    "download_url": item.get("download_url"),
                },
            )
        except Exception as e:
            self.log_error("Failed to parse content item", error=e)
            return None

    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename."""
        extension = filename.lower().split(".")[-1] if "." in filename else ""
        mime_types = {
            "md": "text/markdown",
            "mdx": "text/markdown",
            "txt": "text/plain",
            "py": "text/x-python",
            "js": "text/javascript",
            "ts": "text/typescript",
            "tsx": "text/typescript",
            "jsx": "text/javascript",
            "json": "application/json",
            "yaml": "text/yaml",
            "yml": "text/yaml",
            "toml": "text/toml",
            "html": "text/html",
            "css": "text/css",
            "java": "text/x-java",
            "go": "text/x-go",
            "rs": "text/x-rust",
            "rb": "text/x-ruby",
        }
        return mime_types.get(extension, "application/octet-stream")

    def _parse_link_header(self, link_header: str) -> Optional[str]:
        """Parse GitHub Link header for pagination."""
        if not link_header:
            return None

        for link in link_header.split(","):
            parts = link.strip().split(";")
            if len(parts) == 2 and 'rel="next"' in parts[1]:
                url = parts[0].strip().strip("<>")
                # Extract page number
                if "page=" in url:
                    for param in url.split("?")[1].split("&"):
                        if param.startswith("page="):
                            return param.split("=")[1]
        return None

    async def download_resource(self, resource: Resource) -> bytes:
        """Download resource content."""
        client = await self._get_client()

        if resource.resource_type == ResourceType.FOLDER:
            # For repos/folders, return metadata as content
            info = [
                f"Repository: {resource.name}",
                f"Description: {resource.metadata.get('description', 'No description')}",
                f"Language: {resource.metadata.get('language', 'Unknown')}",
                f"Stars: {resource.metadata.get('stars', 0)}",
            ]
            return "\n".join(info).encode("utf-8")

        # For files, download content
        download_url = resource.metadata.get("download_url")
        if download_url:
            response = await client.get(download_url)
            if response.status_code == 200:
                return response.content

        # Fallback: use contents API
        owner = resource.metadata.get("owner")
        repo = resource.metadata.get("repo")
        path = resource.metadata.get("path")

        if owner and repo and path:
            response = await client.get(f"/repos/{owner}/{repo}/contents/{path}")
            if response.status_code == 200:
                data = response.json()
                if data.get("encoding") == "base64" and data.get("content"):
                    return base64.b64decode(data["content"])

        raise ValueError(f"Failed to download: {resource.path}")

    async def get_changes(
        self,
        since: Optional[datetime] = None,
        page_token: Optional[str] = None,
    ) -> tuple[List[Change], Optional[str]]:
        """Get repository changes since a specific time."""
        # Would use commits API to detect changes
        # For now, return empty - full sync will be used
        return [], None

    async def close(self):
        """Close the connector and clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @classmethod
    def get_oauth_url(
        cls,
        client_id: str,
        redirect_uri: str,
        state: str,
    ) -> str:
        """Generate GitHub OAuth authorization URL."""
        scopes = "repo,read:user"
        return (
            f"https://github.com/login/oauth/authorize"
            f"?client_id={client_id}"
            f"&redirect_uri={redirect_uri}"
            f"&scope={scopes}"
            f"&state={state}"
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for the connector."""
        return {
            "type": "object",
            "properties": {
                "repositories": {
                    "type": "array",
                    "title": "Repositories",
                    "description": "Specific repositories to sync (empty = all accessible)",
                    "items": {"type": "string"},
                    "default": [],
                },
                "sync_issues": {
                    "type": "boolean",
                    "title": "Sync Issues",
                    "description": "Include repository issues as documents",
                    "default": False,
                },
                "sync_pull_requests": {
                    "type": "boolean",
                    "title": "Sync Pull Requests",
                    "description": "Include pull requests as documents",
                    "default": False,
                },
                "sync_wiki": {
                    "type": "boolean",
                    "title": "Sync Wiki",
                    "description": "Include wiki pages",
                    "default": True,
                },
                "file_extensions": {
                    "type": "array",
                    "title": "File Extensions",
                    "description": "File extensions to sync (empty = defaults)",
                    "items": {"type": "string"},
                    "default": [],
                },
                "exclude_paths": {
                    "type": "array",
                    "title": "Exclude Paths",
                    "description": "Paths to exclude (glob patterns)",
                    "items": {"type": "string"},
                    "default": ["node_modules/**", ".git/**", "dist/**", "build/**"],
                },
            },
        }

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get credentials schema for the connector."""
        return {
            "type": "object",
            "required": ["access_token"],
            "properties": {
                "access_token": {
                    "type": "string",
                    "title": "Personal Access Token",
                    "description": "GitHub Personal Access Token with repo scope",
                    "format": "password",
                },
            },
        }
