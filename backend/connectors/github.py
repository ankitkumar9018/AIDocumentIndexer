"""
GitHub Connector for AIDocumentIndexer
======================================

Fetches repositories, code files, issues, and documentation from GitHub.
Supports:
- Repository content indexing
- README and documentation files
- Issues and discussions
- Code search
- Incremental sync via webhooks or polling
"""

import asyncio
import base64
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GitHubFile(BaseModel):
    """Represents a file from GitHub."""
    path: str
    name: str
    content: str
    sha: str
    size: int
    url: str
    html_url: str
    repository: str
    branch: str
    encoding: str = "utf-8"
    language: Optional[str] = None
    last_modified: Optional[datetime] = None

    @property
    def extension(self) -> str:
        return Path(self.path).suffix.lower()

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


class GitHubIssue(BaseModel):
    """Represents a GitHub issue or PR."""
    number: int
    title: str
    body: str
    state: str  # open, closed
    url: str
    html_url: str
    repository: str
    created_at: datetime
    updated_at: datetime
    author: str
    labels: list[str] = Field(default_factory=list)
    is_pull_request: bool = False
    comments_count: int = 0


class GitHubRepository(BaseModel):
    """Represents a GitHub repository."""
    full_name: str  # owner/repo
    name: str
    description: Optional[str]
    url: str
    html_url: str
    default_branch: str
    language: Optional[str]
    stars: int
    forks: int
    open_issues: int
    topics: list[str] = Field(default_factory=list)
    updated_at: datetime
    is_private: bool = False


class GitHubConnectorConfig(BaseModel):
    """Configuration for GitHub connector."""
    access_token: str
    repositories: list[str] = Field(default_factory=list)  # owner/repo format
    organizations: list[str] = Field(default_factory=list)
    include_code: bool = True
    include_issues: bool = True
    include_prs: bool = True
    include_discussions: bool = False
    code_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".rst", ".txt",
            ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp",
            ".rb", ".php", ".swift", ".kt", ".scala",
            ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg",
            ".sh", ".bash", ".zsh",
            ".html", ".css", ".scss", ".less",
            ".sql", ".graphql",
        ]
    )
    excluded_paths: list[str] = Field(
        default_factory=lambda: [
            "node_modules/", "vendor/", "venv/", ".git/",
            "dist/", "build/", "__pycache__/", ".cache/",
            "*.min.js", "*.min.css", "*.lock",
        ]
    )
    max_file_size_kb: int = 500
    max_files_per_repo: int = 1000
    sync_interval_minutes: int = 60


class GitHubConnector:
    """
    GitHub API connector for fetching repository content.

    Usage:
        connector = GitHubConnector(access_token="ghp_xxx")
        async for file in connector.fetch_repository_files("owner/repo"):
            print(f"Fetched: {file.path}")
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, config: GitHubConnectorConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limit_remaining: int = 5000
        self._rate_limit_reset: Optional[datetime] = None

    @property
    def headers(self) -> dict:
        """Get API headers."""
        return {
            "Authorization": f"Bearer {self.config.access_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers=self.headers,
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Optional[dict | list]:
        """Make a rate-limited API request."""
        # Check rate limit
        if self._rate_limit_remaining <= 1 and self._rate_limit_reset:
            wait_time = (self._rate_limit_reset - datetime.now(timezone.utc)).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limited, waiting {wait_time:.0f}s")
                await asyncio.sleep(min(wait_time, 60))

        client = await self._get_client()
        try:
            response = await client.request(method, endpoint, **kwargs)

            # Update rate limit info
            self._rate_limit_remaining = int(
                response.headers.get("x-ratelimit-remaining", 5000)
            )
            reset_timestamp = response.headers.get("x-ratelimit-reset")
            if reset_timestamp:
                self._rate_limit_reset = datetime.fromtimestamp(
                    int(reset_timestamp), tz=timezone.utc
                )

            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.error("GitHub API rate limit exceeded")
            elif e.response.status_code == 404:
                logger.warning(f"Resource not found: {endpoint}")
            else:
                logger.error(f"GitHub API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None

    async def test_connection(self) -> bool:
        """Test the API connection."""
        result = await self._request("GET", "/user")
        return result is not None

    async def get_user_info(self) -> Optional[dict]:
        """Get authenticated user info."""
        return await self._request("GET", "/user")

    # =========================================================================
    # Repository Operations
    # =========================================================================

    async def get_repository(self, repo: str) -> Optional[GitHubRepository]:
        """Get repository details."""
        data = await self._request("GET", f"/repos/{repo}")
        if not data:
            return None

        return GitHubRepository(
            full_name=data["full_name"],
            name=data["name"],
            description=data.get("description"),
            url=data["url"],
            html_url=data["html_url"],
            default_branch=data["default_branch"],
            language=data.get("language"),
            stars=data["stargazers_count"],
            forks=data["forks_count"],
            open_issues=data["open_issues_count"],
            topics=data.get("topics", []),
            updated_at=datetime.fromisoformat(
                data["updated_at"].replace("Z", "+00:00")
            ),
            is_private=data["private"],
        )

    async def list_user_repositories(
        self,
        visibility: str = "all",
        sort: str = "updated",
    ) -> AsyncGenerator[GitHubRepository, None]:
        """List repositories for authenticated user."""
        page = 1
        per_page = 100

        while True:
            data = await self._request(
                "GET",
                "/user/repos",
                params={
                    "visibility": visibility,
                    "sort": sort,
                    "per_page": per_page,
                    "page": page,
                },
            )

            if not data:
                break

            for repo in data:
                yield GitHubRepository(
                    full_name=repo["full_name"],
                    name=repo["name"],
                    description=repo.get("description"),
                    url=repo["url"],
                    html_url=repo["html_url"],
                    default_branch=repo["default_branch"],
                    language=repo.get("language"),
                    stars=repo["stargazers_count"],
                    forks=repo["forks_count"],
                    open_issues=repo["open_issues_count"],
                    topics=repo.get("topics", []),
                    updated_at=datetime.fromisoformat(
                        repo["updated_at"].replace("Z", "+00:00")
                    ),
                    is_private=repo["private"],
                )

            if len(data) < per_page:
                break

            page += 1
            await asyncio.sleep(0.1)

    async def list_org_repositories(
        self,
        org: str,
    ) -> AsyncGenerator[GitHubRepository, None]:
        """List repositories for an organization."""
        page = 1
        per_page = 100

        while True:
            data = await self._request(
                "GET",
                f"/orgs/{org}/repos",
                params={"per_page": per_page, "page": page},
            )

            if not data:
                break

            for repo in data:
                yield GitHubRepository(
                    full_name=repo["full_name"],
                    name=repo["name"],
                    description=repo.get("description"),
                    url=repo["url"],
                    html_url=repo["html_url"],
                    default_branch=repo["default_branch"],
                    language=repo.get("language"),
                    stars=repo["stargazers_count"],
                    forks=repo["forks_count"],
                    open_issues=repo["open_issues_count"],
                    topics=repo.get("topics", []),
                    updated_at=datetime.fromisoformat(
                        repo["updated_at"].replace("Z", "+00:00")
                    ),
                    is_private=repo["private"],
                )

            if len(data) < per_page:
                break

            page += 1
            await asyncio.sleep(0.1)

    # =========================================================================
    # File Operations
    # =========================================================================

    async def fetch_repository_files(
        self,
        repo: str,
        branch: Optional[str] = None,
        path: str = "",
    ) -> AsyncGenerator[GitHubFile, None]:
        """
        Fetch all files from a repository.

        Args:
            repo: Repository in owner/repo format
            branch: Branch to fetch (defaults to default branch)
            path: Starting path within the repo

        Yields:
            GitHubFile objects
        """
        # Get repo info for default branch
        if not branch:
            repo_info = await self.get_repository(repo)
            if not repo_info:
                return
            branch = repo_info.default_branch

        file_count = 0
        async for file in self._fetch_tree_files(repo, branch, path):
            if file_count >= self.config.max_files_per_repo:
                logger.info(f"Reached max files limit for {repo}")
                break

            file_count += 1
            yield file

    async def _fetch_tree_files(
        self,
        repo: str,
        branch: str,
        path: str,
    ) -> AsyncGenerator[GitHubFile, None]:
        """Recursively fetch files from repository tree."""
        # Use Git Trees API for efficient fetching
        data = await self._request(
            "GET",
            f"/repos/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        )

        if not data or "tree" not in data:
            return

        for item in data["tree"]:
            if item["type"] != "blob":
                continue

            file_path = item["path"]

            # Check path filter
            if path and not file_path.startswith(path):
                continue

            # Check exclusions
            if self._should_exclude(file_path):
                continue

            # Check extension
            ext = Path(file_path).suffix.lower()
            if ext not in self.config.code_extensions:
                continue

            # Check size
            size_kb = item.get("size", 0) / 1024
            if size_kb > self.config.max_file_size_kb:
                continue

            # Fetch file content
            file = await self._fetch_file_content(repo, branch, file_path, item["sha"])
            if file:
                yield file

            await asyncio.sleep(0.05)

    async def _fetch_file_content(
        self,
        repo: str,
        branch: str,
        path: str,
        sha: str,
    ) -> Optional[GitHubFile]:
        """Fetch content of a single file."""
        data = await self._request(
            "GET",
            f"/repos/{repo}/contents/{path}",
            params={"ref": branch},
        )

        if not data:
            return None

        # Decode base64 content
        try:
            content = base64.b64decode(data.get("content", "")).decode("utf-8")
        except Exception:
            return None

        # Detect language
        language = self._detect_language(path)

        return GitHubFile(
            path=path,
            name=data["name"],
            content=content,
            sha=sha,
            size=data.get("size", 0),
            url=data["url"],
            html_url=data["html_url"],
            repository=repo,
            branch=branch,
            language=language,
        )

    def _should_exclude(self, path: str) -> bool:
        """Check if path should be excluded."""
        path_lower = path.lower()
        for pattern in self.config.excluded_paths:
            if pattern.endswith("/"):
                if pattern[:-1] in path_lower:
                    return True
            elif pattern.startswith("*."):
                if path_lower.endswith(pattern[1:]):
                    return True
            elif pattern in path_lower:
                return True
        return False

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript",
            ".jsx": "JavaScript",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".cpp": "C++",
            ".c": "C",
            ".h": "C",
            ".hpp": "C++",
            ".rb": "Ruby",
            ".php": "PHP",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".md": "Markdown",
            ".rst": "reStructuredText",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".json": "JSON",
            ".toml": "TOML",
            ".sql": "SQL",
            ".html": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".sh": "Shell",
            ".bash": "Shell",
        }
        ext = Path(path).suffix.lower()
        return ext_to_lang.get(ext, "Text")

    # =========================================================================
    # Issues and PRs
    # =========================================================================

    async def fetch_issues(
        self,
        repo: str,
        state: str = "all",
        since: Optional[datetime] = None,
    ) -> AsyncGenerator[GitHubIssue, None]:
        """Fetch issues from a repository."""
        page = 1
        per_page = 100

        params: dict[str, Any] = {
            "state": state,
            "per_page": per_page,
        }
        if since:
            params["since"] = since.isoformat()

        while True:
            params["page"] = page
            data = await self._request(
                "GET",
                f"/repos/{repo}/issues",
                params=params,
            )

            if not data:
                break

            for issue in data:
                is_pr = "pull_request" in issue

                # Skip PRs if not configured
                if is_pr and not self.config.include_prs:
                    continue
                if not is_pr and not self.config.include_issues:
                    continue

                yield GitHubIssue(
                    number=issue["number"],
                    title=issue["title"],
                    body=issue.get("body") or "",
                    state=issue["state"],
                    url=issue["url"],
                    html_url=issue["html_url"],
                    repository=repo,
                    created_at=datetime.fromisoformat(
                        issue["created_at"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        issue["updated_at"].replace("Z", "+00:00")
                    ),
                    author=issue["user"]["login"],
                    labels=[l["name"] for l in issue.get("labels", [])],
                    is_pull_request=is_pr,
                    comments_count=issue.get("comments", 0),
                )

            if len(data) < per_page:
                break

            page += 1
            await asyncio.sleep(0.1)

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search_code(
        self,
        query: str,
        repo: Optional[str] = None,
        language: Optional[str] = None,
        max_results: int = 100,
    ) -> AsyncGenerator[GitHubFile, None]:
        """Search code across repositories."""
        search_query = query

        if repo:
            search_query += f" repo:{repo}"
        if language:
            search_query += f" language:{language}"

        page = 1
        per_page = min(100, max_results)
        total_fetched = 0

        while total_fetched < max_results:
            data = await self._request(
                "GET",
                "/search/code",
                params={
                    "q": search_query,
                    "per_page": per_page,
                    "page": page,
                },
            )

            if not data or "items" not in data:
                break

            for item in data["items"]:
                if total_fetched >= max_results:
                    break

                # Fetch full file content
                file = await self._fetch_file_content(
                    item["repository"]["full_name"],
                    item["repository"]["default_branch"],
                    item["path"],
                    item["sha"],
                )

                if file:
                    total_fetched += 1
                    yield file

                await asyncio.sleep(0.1)

            if len(data["items"]) < per_page:
                break

            page += 1

    # =========================================================================
    # Full Sync
    # =========================================================================

    async def sync_all(self) -> AsyncGenerator[GitHubFile | GitHubIssue, None]:
        """
        Sync all configured repositories.

        Yields:
            GitHubFile and GitHubIssue objects
        """
        # Sync configured repos
        for repo in self.config.repositories:
            logger.info(f"Syncing repository: {repo}")

            if self.config.include_code:
                async for file in self.fetch_repository_files(repo):
                    yield file

            if self.config.include_issues or self.config.include_prs:
                async for issue in self.fetch_issues(repo):
                    yield issue

        # Sync organization repos
        for org in self.config.organizations:
            logger.info(f"Syncing organization: {org}")

            async for repo in self.list_org_repositories(org):
                if self.config.include_code:
                    async for file in self.fetch_repository_files(repo.full_name):
                        yield file

                if self.config.include_issues or self.config.include_prs:
                    async for issue in self.fetch_issues(repo.full_name):
                        yield issue


# Factory function
def create_github_connector(
    access_token: str,
    repositories: list[str] = None,
    organizations: list[str] = None,
    include_code: bool = True,
    include_issues: bool = True,
    include_prs: bool = True,
) -> GitHubConnector:
    """
    Create a GitHub connector instance.

    Args:
        access_token: GitHub personal access token
        repositories: List of repositories (owner/repo format)
        organizations: List of organizations to sync
        include_code: Whether to index code files
        include_issues: Whether to index issues
        include_prs: Whether to index pull requests

    Returns:
        Configured GitHubConnector instance
    """
    config = GitHubConnectorConfig(
        access_token=access_token,
        repositories=repositories or [],
        organizations=organizations or [],
        include_code=include_code,
        include_issues=include_issues,
        include_prs=include_prs,
    )
    return GitHubConnector(config)
