"""
AIDocumentIndexer Connectors
============================

External service connectors for importing documents from various sources.
"""

from .notion import (
    NotionConnector,
    NotionConnectorConfig,
    NotionPage,
    NotionDatabase,
    create_notion_connector,
)
from .github import (
    GitHubConnector,
    GitHubConnectorConfig,
    GitHubFile,
    GitHubIssue,
    GitHubRepository,
    create_github_connector,
)

__all__ = [
    # Notion
    "NotionConnector",
    "NotionConnectorConfig",
    "NotionPage",
    "NotionDatabase",
    "create_notion_connector",
    # GitHub
    "GitHubConnector",
    "GitHubConnectorConfig",
    "GitHubFile",
    "GitHubIssue",
    "GitHubRepository",
    "create_github_connector",
]
