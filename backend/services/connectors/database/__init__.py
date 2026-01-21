"""
AIDocumentIndexer - Database Connectors
========================================

Database connectors for natural language querying of external databases.
Supports PostgreSQL, MySQL, MongoDB, and SQLite.
"""

from backend.services.connectors.database.base import (
    BaseDatabaseConnector,
    DatabaseConnectorType,
    DatabaseSchema,
    TableSchema,
    ColumnSchema,
    QueryResult,
    QueryValidationResult,
)
from backend.services.connectors.database.postgresql import PostgreSQLConnector
from backend.services.connectors.database.mysql import MySQLConnector
from backend.services.connectors.database.mongodb import MongoDBConnector
from backend.services.connectors.database.sqlite import SQLiteConnector

__all__ = [
    # Base classes
    "BaseDatabaseConnector",
    "DatabaseConnectorType",
    "DatabaseSchema",
    "TableSchema",
    "ColumnSchema",
    "QueryResult",
    "QueryValidationResult",
    # Connectors
    "PostgreSQLConnector",
    "MySQLConnector",
    "MongoDBConnector",
    "SQLiteConnector",
]
