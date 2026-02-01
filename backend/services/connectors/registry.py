"""
AIDocumentIndexer - Connector Registry
=======================================

Central registry for all available connectors.
Provides discovery, instantiation, and management of connectors.
"""

from typing import Dict, Type, List, Optional, Any

import structlog

from backend.services.connectors.base import BaseConnector, ConnectorType, ConnectorConfig

logger = structlog.get_logger(__name__)


class ConnectorRegistry:
    """
    Registry for connector types.

    Maintains a mapping of connector types to their implementations
    and provides factory methods for instantiation.
    """

    _connectors: Dict[ConnectorType, Type[BaseConnector]] = {}

    @classmethod
    def register(cls, connector_type: ConnectorType):
        """
        Decorator to register a connector class.

        Usage:
            @ConnectorRegistry.register(ConnectorType.GOOGLE_DRIVE)
            class GoogleDriveConnector(BaseConnector):
                ...
        """
        def decorator(connector_class: Type[BaseConnector]):
            cls._connectors[connector_type] = connector_class
            logger.debug(f"Registered connector: {connector_type.value}")
            return connector_class
        return decorator

    @classmethod
    def get(cls, connector_type: ConnectorType) -> Optional[Type[BaseConnector]]:
        """Get a connector class by type."""
        return cls._connectors.get(connector_type)

    @classmethod
    def list_all(cls) -> List[Dict[str, Any]]:
        """
        List all registered connectors with their metadata.

        Returns:
            List of connector info dicts
        """
        connectors = []

        for conn_type, conn_class in cls._connectors.items():
            connectors.append({
                "type": conn_type.value,
                "name": conn_class.display_name,
                "description": conn_class.description,
                "icon": conn_class.icon,
                "config_schema": conn_class.get_config_schema(),
                "credentials_schema": conn_class.get_credentials_schema(),
            })

        return connectors

    @classmethod
    def create(
        cls,
        connector_type: ConnectorType,
        config: ConnectorConfig,
        session=None,
        organization_id=None,
        user_id=None,
    ) -> Optional[BaseConnector]:
        """
        Create a connector instance.

        Args:
            connector_type: Type of connector
            config: Connector configuration
            session: Database session
            organization_id: Organization context
            user_id: User context

        Returns:
            Connector instance or None if type not found
        """
        connector_class = cls.get(connector_type)

        if not connector_class:
            logger.warning(f"Unknown connector type: {connector_type}")
            return None

        return connector_class(
            config=config,
            session=session,
            organization_id=organization_id,
            user_id=user_id,
        )


def get_connector(
    connector_type: str,
    config: Dict[str, Any],
    session=None,
    organization_id=None,
    user_id=None,
) -> Optional[BaseConnector]:
    """
    Convenience function to create a connector.

    Args:
        connector_type: Connector type as string
        config: Configuration dict
        session: Database session
        organization_id: Organization context
        user_id: User context

    Returns:
        Connector instance or None
    """
    try:
        conn_type = ConnectorType(connector_type)
    except ValueError:
        logger.warning(f"Invalid connector type: {connector_type}")
        return None

    connector_config = ConnectorConfig(
        connector_type=conn_type,
        **config,
    )

    return ConnectorRegistry.create(
        connector_type=conn_type,
        config=connector_config,
        session=session,
        organization_id=organization_id,
        user_id=user_id,
    )


# Import connectors to register them
def _register_builtin_connectors():
    """Import and register all built-in connectors."""
    try:
        from backend.services.connectors.google_drive import GoogleDriveConnector
    except ImportError as e:
        logger.warning(f"Failed to import Google Drive connector: {e}")

    try:
        from backend.services.connectors.notion import NotionConnector
    except ImportError as e:
        logger.debug(f"Notion connector not available: {e}")

    try:
        from backend.services.connectors.confluence import ConfluenceConnector
    except ImportError as e:
        logger.debug(f"Confluence connector not available: {e}")

    try:
        from backend.services.connectors.onedrive import OneDriveConnector, SharePointConnector
    except ImportError as e:
        logger.debug(f"OneDrive/SharePoint connector not available: {e}")

    try:
        from backend.services.connectors.youtube import YouTubeConnector
    except ImportError as e:
        logger.debug(f"YouTube connector not available: {e}")

    try:
        from backend.services.connectors.slack_data import SlackDataConnector
    except ImportError as e:
        logger.debug(f"Slack data connector not available: {e}")

    try:
        from backend.services.connectors.github import GitHubConnector
    except ImportError as e:
        logger.debug(f"GitHub connector not available: {e}")

    try:
        from backend.services.connectors.dropbox import DropboxConnector
    except ImportError as e:
        logger.debug(f"Dropbox connector not available: {e}")

    try:
        from backend.services.connectors.box import BoxConnector
    except ImportError as e:
        logger.debug(f"Box connector not available: {e}")


# Register connectors on module load
_register_builtin_connectors()
