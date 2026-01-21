"""
AIDocumentIndexer - Base Database Connector
============================================

Abstract base class for database connectors that enable natural language
querying of external databases.

Features:
- Schema introspection
- Read-only query execution
- SQL validation and safety checks
- Connection pooling
- Sample data retrieval for few-shot prompting
"""

import hashlib
import re
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class DatabaseConnectorType(str, Enum):
    """Supported database connector types."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    SQLITE = "sqlite"


class ColumnSchema(BaseModel):
    """Schema for a database column."""
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    default_value: Optional[str] = None
    description: Optional[str] = None
    sample_values: List[Any] = Field(default_factory=list)


class TableSchema(BaseModel):
    """Schema for a database table."""
    name: str
    schema_name: Optional[str] = None  # e.g., "public" for PostgreSQL
    columns: List[ColumnSchema] = Field(default_factory=list)
    primary_key: Optional[List[str]] = None
    foreign_keys: List[Dict[str, Any]] = Field(default_factory=list)
    indexes: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: Optional[int] = None
    description: Optional[str] = None


class DatabaseSchema(BaseModel):
    """Complete schema for a database."""
    database_name: str
    connector_type: DatabaseConnectorType
    tables: List[TableSchema] = Field(default_factory=list)
    views: List[TableSchema] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def get_table(self, name: str) -> Optional[TableSchema]:
        """Get a table by name."""
        for table in self.tables:
            if table.name.lower() == name.lower():
                return table
        return None

    def get_column_names(self, table_name: str) -> List[str]:
        """Get column names for a table."""
        table = self.get_table(table_name)
        if table:
            return [col.name for col in table.columns]
        return []

    def to_ddl_string(self, include_sample_values: bool = True) -> str:
        """Convert schema to DDL string for LLM context."""
        ddl_parts = []

        for table in self.tables:
            columns_ddl = []
            for col in table.columns:
                col_def = f"  {col.name} {col.data_type}"
                if col.is_primary_key:
                    col_def += " PRIMARY KEY"
                if not col.is_nullable:
                    col_def += " NOT NULL"
                if col.is_foreign_key and col.foreign_key_table:
                    col_def += f" REFERENCES {col.foreign_key_table}({col.foreign_key_column})"
                columns_ddl.append(col_def)

            table_ddl = f"CREATE TABLE {table.name} (\n"
            table_ddl += ",\n".join(columns_ddl)
            table_ddl += "\n);"

            if include_sample_values and table.columns:
                # Add sample values as comments
                sample_comment = f"\n-- Sample values for {table.name}:"
                for col in table.columns[:5]:  # Limit to first 5 columns
                    if col.sample_values:
                        samples = ", ".join(str(v) for v in col.sample_values[:3])
                        sample_comment += f"\n--   {col.name}: {samples}"
                table_ddl += sample_comment

            ddl_parts.append(table_ddl)

        return "\n\n".join(ddl_parts)


class QueryResult(BaseModel):
    """Result of a database query."""
    success: bool
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0
    error: Optional[str] = None
    query: Optional[str] = None
    truncated: bool = False  # True if results were limited

    def to_dict_rows(self) -> List[Dict[str, Any]]:
        """Convert rows to list of dicts."""
        return [dict(zip(self.columns, row)) for row in self.rows]


class QueryValidationResult(BaseModel):
    """Result of SQL query validation."""
    is_valid: bool
    is_read_only: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    tables_referenced: List[str] = Field(default_factory=list)
    query_type: Optional[str] = None  # SELECT, INSERT, UPDATE, DELETE, etc.


# Dangerous SQL keywords that should be blocked
DANGEROUS_KEYWORDS = {
    # Data modification
    "INSERT", "UPDATE", "DELETE", "TRUNCATE", "DROP", "ALTER", "CREATE",
    "REPLACE", "MERGE", "UPSERT",
    # Schema modification
    "GRANT", "REVOKE", "RENAME",
    # Transaction control that could be misused
    "COMMIT", "ROLLBACK", "SAVEPOINT",
    # Database admin
    "VACUUM", "ANALYZE", "REINDEX", "CLUSTER",
    # File operations
    "COPY", "LOAD", "INTO OUTFILE", "INTO DUMPFILE",
    # Dangerous functions
    "EXEC", "EXECUTE", "CALL", "XP_",
}

# Patterns that indicate dangerous queries
DANGEROUS_PATTERNS = [
    r";\s*(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)",  # Chained statements
    r"--\s*$",  # Line comments that could hide malicious code
    r"/\*.*?\*/",  # Block comments
    r"UNION\s+ALL\s+SELECT.*FROM\s+information_schema",  # Schema enumeration
    r"BENCHMARK\s*\(",  # Time-based attacks
    r"SLEEP\s*\(",  # Time-based attacks
    r"pg_sleep\s*\(",  # PostgreSQL time-based attacks
    r"WAITFOR\s+DELAY",  # SQL Server time-based attacks
]


class BaseDatabaseConnector(ABC):
    """
    Abstract base class for database connectors.

    Provides a unified interface for:
    - Connection management
    - Schema introspection
    - Read-only query execution
    - SQL validation
    """

    connector_type: DatabaseConnectorType = None
    display_name: str = "Base Database Connector"
    description: str = "Abstract base database connector"

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
        ssl_mode: Optional[str] = None,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_rows: int = 1000,
        query_timeout_seconds: int = 30,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.ssl_mode = ssl_mode
        self.organization_id = organization_id
        self.user_id = user_id
        self.max_rows = max_rows
        self.query_timeout_seconds = query_timeout_seconds

        self._connected = False
        self._schema_cache: Optional[DatabaseSchema] = None
        self._connection = None

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._connected

    @property
    def connection_string_hash(self) -> str:
        """Get a hash of the connection string for caching."""
        conn_str = f"{self.host}:{self.port}/{self.database}/{self.username}"
        return hashlib.sha256(conn_str.encode()).hexdigest()[:16]

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the database.

        Returns:
            True if connection successful

        Raises:
            ConnectionError if connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection."""
        pass

    @abstractmethod
    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Test database connectivity.

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        pass

    @abstractmethod
    async def get_schema(self, refresh: bool = False) -> DatabaseSchema:
        """
        Get the database schema.

        Args:
            refresh: If True, refresh the cached schema

        Returns:
            DatabaseSchema object
        """
        pass

    @abstractmethod
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a read-only SQL query.

        Args:
            query: SQL query string
            parameters: Query parameters for parameterized queries

        Returns:
            QueryResult with data or error
        """
        pass

    @abstractmethod
    async def get_sample_data(
        self,
        table_name: str,
        limit: int = 5,
    ) -> QueryResult:
        """
        Get sample data from a table for few-shot examples.

        Args:
            table_name: Name of the table
            limit: Number of rows to retrieve

        Returns:
            QueryResult with sample data
        """
        pass

    async def get_table_names(self) -> List[str]:
        """Get list of table names."""
        schema = await self.get_schema()
        return [table.name for table in schema.tables]

    async def get_column_info(self, table_name: str) -> List[ColumnSchema]:
        """Get column information for a table."""
        schema = await self.get_schema()
        table = schema.get_table(table_name)
        return table.columns if table else []

    def validate_query(self, query: str) -> QueryValidationResult:
        """
        Validate a SQL query for safety.

        Checks:
        - Query is read-only (SELECT, WITH...SELECT only)
        - No dangerous keywords
        - No suspicious patterns

        Args:
            query: SQL query to validate

        Returns:
            QueryValidationResult
        """
        result = QueryValidationResult(
            is_valid=True,
            is_read_only=True,
            errors=[],
            warnings=[],
            tables_referenced=[],
        )

        # Normalize query for analysis
        normalized = query.upper().strip()

        # Remove string literals to avoid false positives
        # Replace quoted strings with placeholders
        normalized_clean = re.sub(r"'[^']*'", "''", normalized)
        normalized_clean = re.sub(r'"[^"]*"', '""', normalized_clean)

        # Check for multiple statements (except WITH...SELECT)
        statements = [s.strip() for s in normalized_clean.split(';') if s.strip()]
        if len(statements) > 1:
            result.is_valid = False
            result.is_read_only = False
            result.errors.append("Multiple statements not allowed")

        # Determine query type
        if normalized_clean.startswith("SELECT"):
            result.query_type = "SELECT"
        elif normalized_clean.startswith("WITH"):
            result.query_type = "WITH"
            # Check that WITH ends with SELECT
            if " SELECT " not in normalized_clean:
                result.is_valid = False
                result.errors.append("WITH clause must contain SELECT")
        elif normalized_clean.startswith("EXPLAIN"):
            result.query_type = "EXPLAIN"
            result.warnings.append("EXPLAIN queries may expose query plans")
        else:
            result.is_valid = False
            result.is_read_only = False
            result.query_type = normalized_clean.split()[0] if normalized_clean else "UNKNOWN"
            result.errors.append(f"Only SELECT queries allowed, got: {result.query_type}")

        # Check for dangerous keywords
        for keyword in DANGEROUS_KEYWORDS:
            # Use word boundary to avoid false positives like "UPDATED_AT"
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, normalized_clean):
                result.is_valid = False
                result.is_read_only = False
                result.errors.append(f"Dangerous keyword found: {keyword}")

        # Check for dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                result.is_valid = False
                result.warnings.append(f"Suspicious pattern detected")

        # Extract referenced tables (basic extraction)
        # This is a simplified extraction - each connector can override for accuracy
        from_match = re.findall(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', normalized_clean)
        join_match = re.findall(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', normalized_clean)
        result.tables_referenced = list(set(from_match + join_match))

        return result

    async def execute_validated_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Validate and execute a query.

        Args:
            query: SQL query string
            parameters: Query parameters

        Returns:
            QueryResult with data or validation error
        """
        # Validate first
        validation = self.validate_query(query)

        if not validation.is_valid:
            return QueryResult(
                success=False,
                error=f"Query validation failed: {'; '.join(validation.errors)}",
                query=query,
            )

        if not validation.is_read_only:
            return QueryResult(
                success=False,
                error="Only read-only queries are allowed",
                query=query,
            )

        # Execute the query
        return await self.execute_query(query, parameters)

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """
        Get JSON schema for connector credentials.

        Returns:
            JSON schema dict
        """
        return {
            "type": "object",
            "required": ["host", "port", "database", "username", "password"],
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Database host address",
                },
                "port": {
                    "type": "integer",
                    "description": "Database port number",
                },
                "database": {
                    "type": "string",
                    "description": "Database name",
                },
                "username": {
                    "type": "string",
                    "description": "Database username",
                },
                "password": {
                    "type": "string",
                    "format": "password",
                    "description": "Database password",
                },
                "ssl_mode": {
                    "type": "string",
                    "enum": ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"],
                    "default": "prefer",
                    "description": "SSL connection mode",
                },
            },
        }

    def log_info(self, message: str, **kwargs):
        """Log info message with context."""
        logger.info(
            message,
            connector_type=self.connector_type.value if self.connector_type else None,
            database=self.database,
            **kwargs,
        )

    def log_error(self, message: str, **kwargs):
        """Log error message with context."""
        logger.error(
            message,
            connector_type=self.connector_type.value if self.connector_type else None,
            database=self.database,
            **kwargs,
        )

    def log_warning(self, message: str, **kwargs):
        """Log warning message with context."""
        logger.warning(
            message,
            connector_type=self.connector_type.value if self.connector_type else None,
            database=self.database,
            **kwargs,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
