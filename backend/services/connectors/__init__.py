"""
AIDocumentIndexer - Connector Framework
========================================

Connect to external data sources and automatically sync documents.

Supported Connectors:
- Google Drive
- Notion
- Confluence
- OneDrive/SharePoint
- Slack (messages and files)
- YouTube (transcription)
"""

from backend.services.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    ConnectorType,
    Resource,
    ResourceType,
    Change,
)
from backend.services.connectors.registry import ConnectorRegistry, get_connector
from backend.services.connectors.scheduler import (
    ConnectorSyncScheduler,
    ConnectorSyncService,
    SyncJob,
    SyncStatus,
    get_scheduler,
    start_scheduler,
    stop_scheduler,
)
from backend.services.connectors.google_drive import GoogleDriveConnector

__all__ = [
    # Base classes
    "BaseConnector",
    "ConnectorConfig",
    "ConnectorType",
    "Resource",
    "ResourceType",
    "Change",
    # Registry
    "ConnectorRegistry",
    "get_connector",
    # Scheduler
    "ConnectorSyncScheduler",
    "ConnectorSyncService",
    "SyncJob",
    "SyncStatus",
    "get_scheduler",
    "start_scheduler",
    "stop_scheduler",
    # Connectors
    "GoogleDriveConnector",
]
