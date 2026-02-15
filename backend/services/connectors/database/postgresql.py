"""
AIDocumentIndexer - PostgreSQL Database Connector
==================================================

PostgreSQL connector for natural language querying using asyncpg.

Features:
- Async connection pooling
- Schema introspection with foreign key detection
- Read-only query execution with statement timeout
- Sample data retrieval for few-shot prompting
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    asyncpg = None
    HAS_ASYNCPG = False

from backend.services.connectors.database.base import (
    BaseDatabaseConnector,
    ColumnSchema,
    DatabaseConnectorType,
    DatabaseSchema,
    QueryResult,
    TableSchema,
)

logger = structlog.get_logger(__name__)


def _sanitize_identifier(name: str) -> str:
    """
    Sanitize a SQL identifier (table/column/schema name) to prevent injection.

    Only allows alphanumeric characters and underscores.
    Raises ValueError for invalid identifiers.
    """
    import re
    if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


class PostgreSQLConnector(BaseDatabaseConnector):
    """
    PostgreSQL database connector using asyncpg.

    Supports:
    - Connection pooling for efficient query execution
    - Full schema introspection including foreign keys
    - Read-only query execution with timeout
    - Sample data retrieval
    """

    connector_type = DatabaseConnectorType.POSTGRESQL
    display_name = "PostgreSQL"
    description = "Connect to PostgreSQL databases for natural language querying"

    def __init__(
        self,
        host: str,
        port: int = 5432,
        database: str = "postgres",
        username: str = "postgres",
        password: str = "",
        ssl_mode: Optional[str] = "prefer",
        schema: str = "public",
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_rows: int = 1000,
        query_timeout_seconds: int = 30,
        pool_min_size: int = 1,
        pool_max_size: int = 5,
    ):
        super().__init__(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            ssl_mode=ssl_mode,
            organization_id=organization_id,
            user_id=user_id,
            max_rows=max_rows,
            query_timeout_seconds=query_timeout_seconds,
        )
        self.schema = schema
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> bool:
        """Establish connection pool to PostgreSQL."""
        if not HAS_ASYNCPG:
            raise ImportError(
                "asyncpg is required for PostgreSQL connections. "
                "Install with: pip install asyncpg"
            )

        try:
            # Build SSL context based on ssl_mode
            ssl = None
            if self.ssl_mode in ("require", "verify-ca", "verify-full"):
                ssl = "require"  # asyncpg handles SSL modes differently

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                ssl=ssl,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=self.query_timeout_seconds,
            )

            self._connected = True
            self.log_info("Connected to PostgreSQL", host=self.host)
            return True

        except Exception as e:
            self.log_error("Failed to connect to PostgreSQL", error=str(e))
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
        self._connected = False
        self.log_info("Disconnected from PostgreSQL")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test database connectivity."""
        try:
            if not self._connected:
                await self.connect()

            async with self._pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    return True, None
                return False, "Unexpected result from test query"

        except Exception as e:
            return False, str(e)

    async def get_schema(self, refresh: bool = False) -> DatabaseSchema:
        """Get the database schema with caching."""
        if self._schema_cache and not refresh:
            return self._schema_cache

        if not self._connected:
            await self.connect()

        tables = []
        views = []

        async with self._pool.acquire() as conn:
            # Get tables
            table_rows = await conn.fetch("""
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = $1
                  AND table_type IN ('BASE TABLE', 'VIEW')
                ORDER BY table_name
            """, self.schema)

            for row in table_rows:
                table_name = row["table_name"]
                is_view = row["table_type"] == "VIEW"

                # Get columns
                columns = await self._get_table_columns(conn, table_name)

                # Merge native column comments from pg_description
                col_comments = await self._get_column_comments(conn, table_name)
                for col in columns:
                    if col.name in col_comments:
                        col.description = col_comments[col.name]

                # Get table-level comment
                table_comment = await self._get_table_comment(conn, table_name)

                # Get primary key
                pk_columns = await self._get_primary_key(conn, table_name)

                # Get foreign keys
                foreign_keys = await self._get_foreign_keys(conn, table_name)

                # Get row count estimate (for tables only)
                row_count = None
                if not is_view:
                    row_count = await self._get_row_count_estimate(conn, table_name)

                table_schema = TableSchema(
                    name=table_name,
                    schema_name=self.schema,
                    columns=columns,
                    primary_key=pk_columns,
                    foreign_keys=foreign_keys,
                    row_count=row_count,
                    description=table_comment,
                )

                if is_view:
                    views.append(table_schema)
                else:
                    tables.append(table_schema)

        self._schema_cache = DatabaseSchema(
            database_name=self.database,
            connector_type=self.connector_type,
            tables=tables,
            views=views,
        )

        return self._schema_cache

    async def _get_table_columns(
        self,
        conn: asyncpg.Connection,
        table_name: str,
    ) -> List[ColumnSchema]:
        """Get columns for a table."""
        columns = []

        rows = await conn.fetch("""
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                fk.foreign_table_name,
                fk.foreign_column_name
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_schema = $1
                  AND tc.table_name = $2
                  AND tc.constraint_type = 'PRIMARY KEY'
            ) pk ON c.column_name = pk.column_name
            LEFT JOIN (
                SELECT
                    kcu.column_name,
                    ccu.table_name as foreign_table_name,
                    ccu.column_name as foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                WHERE tc.table_schema = $1
                  AND tc.table_name = $2
                  AND tc.constraint_type = 'FOREIGN KEY'
            ) fk ON c.column_name = fk.column_name
            WHERE c.table_schema = $1 AND c.table_name = $2
            ORDER BY c.ordinal_position
        """, self.schema, table_name)

        for row in rows:
            # Build data type string with precision info
            data_type = row["data_type"]
            if row["character_maximum_length"]:
                data_type += f"({row['character_maximum_length']})"
            elif row["numeric_precision"] and row["data_type"] in ("numeric", "decimal"):
                scale = row["numeric_scale"] or 0
                data_type += f"({row['numeric_precision']},{scale})"

            columns.append(ColumnSchema(
                name=row["column_name"],
                data_type=data_type,
                is_nullable=row["is_nullable"] == "YES",
                is_primary_key=row["is_primary_key"],
                is_foreign_key=row["foreign_table_name"] is not None,
                foreign_key_table=row["foreign_table_name"],
                foreign_key_column=row["foreign_column_name"],
                default_value=row["column_default"],
            ))

        return columns

    async def _get_primary_key(
        self,
        conn: asyncpg.Connection,
        table_name: str,
    ) -> Optional[List[str]]:
        """Get primary key columns for a table."""
        rows = await conn.fetch("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_schema = $1
              AND tc.table_name = $2
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
        """, self.schema, table_name)

        if rows:
            return [row["column_name"] for row in rows]
        return None

    async def _get_foreign_keys(
        self,
        conn: asyncpg.Connection,
        table_name: str,
    ) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table."""
        rows = await conn.fetch("""
            SELECT
                kcu.column_name,
                ccu.table_name as foreign_table,
                ccu.column_name as foreign_column,
                tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.table_schema = $1
              AND tc.table_name = $2
              AND tc.constraint_type = 'FOREIGN KEY'
        """, self.schema, table_name)

        return [
            {
                "column": row["column_name"],
                "foreign_table": row["foreign_table"],
                "foreign_column": row["foreign_column"],
                "constraint_name": row["constraint_name"],
            }
            for row in rows
        ]

    async def _get_column_comments(
        self,
        conn,
        table_name: str,
    ) -> Dict[str, str]:
        """Fetch column COMMENT metadata from pg_description."""
        try:
            rows = await conn.fetch("""
                SELECT
                    a.attname AS column_name,
                    d.description AS comment
                FROM pg_catalog.pg_attribute a
                JOIN pg_catalog.pg_class c ON a.attrelid = c.oid
                JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                LEFT JOIN pg_catalog.pg_description d
                    ON d.objoid = c.oid AND d.objsubid = a.attnum
                WHERE n.nspname = $1
                  AND c.relname = $2
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                  AND d.description IS NOT NULL
            """, self.schema, table_name)
            return {row["column_name"]: row["comment"] for row in rows}
        except Exception:
            return {}

    async def _get_table_comment(
        self,
        conn,
        table_name: str,
    ) -> Optional[str]:
        """Fetch table-level COMMENT from pg_description."""
        try:
            result = await conn.fetchval("""
                SELECT d.description
                FROM pg_catalog.pg_description d
                JOIN pg_catalog.pg_class c ON d.objoid = c.oid
                JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = $1 AND c.relname = $2 AND d.objsubid = 0
            """, self.schema, table_name)
            return result
        except Exception:
            return None

    async def _get_row_count_estimate(
        self,
        conn: asyncpg.Connection,
        table_name: str,
    ) -> Optional[int]:
        """Get estimated row count from pg_stat."""
        try:
            result = await conn.fetchval("""
                SELECT reltuples::bigint
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = $1 AND c.relname = $2
            """, self.schema, table_name)
            return result if result and result > 0 else None
        except Exception as e:
            self.log_debug("Row count estimate failed", table=table_name, error=str(e))
            return None

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Execute a read-only SQL query."""
        if not self._connected:
            await self.connect()

        start_time = time.time()

        try:
            # Add LIMIT if not present to prevent unbounded queries
            query_upper = query.upper().strip()
            if "LIMIT" not in query_upper and query_upper.startswith(("SELECT", "WITH")):
                query = f"{query.rstrip().rstrip(';')} LIMIT {self.max_rows}"

            async with self._pool.acquire() as conn:
                # Set statement timeout for this transaction
                await conn.execute(
                    f"SET statement_timeout = '{self.query_timeout_seconds * 1000}'"
                )

                # Execute query in a read-only transaction
                async with conn.transaction(readonly=True):
                    if parameters:
                        # Convert dict parameters to positional
                        # asyncpg uses $1, $2, etc. for parameters
                        rows = await conn.fetch(query, *parameters.values())
                    else:
                        rows = await conn.fetch(query)

            # Extract column names and data
            if rows:
                columns = list(rows[0].keys())
                data = [list(row.values()) for row in rows]
                truncated = len(rows) >= self.max_rows
            else:
                columns = []
                data = []
                truncated = False

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                success=True,
                columns=columns,
                rows=data,
                row_count=len(data),
                execution_time_ms=execution_time,
                query=query,
                truncated=truncated,
            )

        except asyncpg.exceptions.QueryCanceledError:
            return QueryResult(
                success=False,
                error=f"Query timed out after {self.query_timeout_seconds} seconds",
                query=query,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            self.log_error("Query execution failed", error=str(e), query=query)
            return QueryResult(
                success=False,
                error=str(e),
                query=query,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def get_sample_data(
        self,
        table_name: str,
        limit: int = 5,
    ) -> QueryResult:
        """Get sample data from a table."""
        # Validate table name to prevent injection
        schema = await self.get_schema()
        if not schema.get_table(table_name):
            return QueryResult(
                success=False,
                error=f"Table '{table_name}' not found",
            )

        # Sanitize identifiers to prevent SQL injection
        safe_schema = _sanitize_identifier(self.schema)
        safe_table = _sanitize_identifier(table_name)
        query = f'SELECT * FROM "{safe_schema}"."{safe_table}" LIMIT {int(limit)}'
        return await self.execute_query(query)

    async def get_distinct_values(
        self,
        table_name: str,
        column_name: str,
        limit: int = 10,
    ) -> List[Any]:
        """Get distinct values for a column (useful for proper noun retrieval)."""
        schema = await self.get_schema()
        table = schema.get_table(table_name)

        if not table:
            return []

        # Verify column exists
        column_names = [col.name for col in table.columns]
        if column_name not in column_names:
            return []

        query = f"""
            SELECT DISTINCT "{column_name}"
            FROM "{self.schema}"."{table_name}"
            WHERE "{column_name}" IS NOT NULL
            LIMIT {limit}
        """

        result = await self.execute_query(query)
        if result.success and result.rows:
            return [row[0] for row in result.rows]
        return []

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for PostgreSQL credentials."""
        base_schema = super().get_credentials_schema()
        base_schema["properties"]["port"]["default"] = 5432
        base_schema["properties"]["schema"] = {
            "type": "string",
            "default": "public",
            "description": "Database schema to use",
        }
        return base_schema
