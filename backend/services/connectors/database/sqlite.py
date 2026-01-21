"""
AIDocumentIndexer - SQLite Database Connector
==============================================

SQLite connector for natural language querying using aiosqlite.

Features:
- Async query execution
- Schema introspection
- Read-only query execution
- Sample data retrieval for few-shot prompting
- File-based database support
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    aiosqlite = None
    HAS_AIOSQLITE = False

from backend.services.connectors.database.base import (
    BaseDatabaseConnector,
    ColumnSchema,
    DatabaseConnectorType,
    DatabaseSchema,
    QueryResult,
    TableSchema,
)

logger = structlog.get_logger(__name__)


class SQLiteConnector(BaseDatabaseConnector):
    """
    SQLite database connector using aiosqlite.

    Supports:
    - File-based SQLite databases
    - In-memory databases (using :memory:)
    - Read-only query execution
    - Schema introspection
    - Sample data retrieval
    """

    connector_type = DatabaseConnectorType.SQLITE
    display_name = "SQLite"
    description = "Connect to SQLite databases for natural language querying"

    def __init__(
        self,
        database_path: str,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_rows: int = 1000,
        query_timeout_seconds: int = 30,
        # SQLite doesn't use these but we accept them for compatibility
        host: str = "",
        port: int = 0,
        database: str = "",
        username: str = "",
        password: str = "",
        ssl_mode: Optional[str] = None,
    ):
        # SQLite uses file path instead of host/port/database
        super().__init__(
            host=host or database_path,
            port=port,
            database=database or database_path,
            username=username,
            password=password,
            ssl_mode=ssl_mode,
            organization_id=organization_id,
            user_id=user_id,
            max_rows=max_rows,
            query_timeout_seconds=query_timeout_seconds,
        )
        self.database_path = database_path
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> bool:
        """Open connection to SQLite database."""
        if not HAS_AIOSQLITE:
            raise ImportError(
                "aiosqlite is required for SQLite connections. "
                "Install with: pip install aiosqlite"
            )

        try:
            self._connection = await aiosqlite.connect(self.database_path)
            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON")
            self._connected = True
            self.log_info("Connected to SQLite", database=self.database_path)
            return True

        except Exception as e:
            self.log_error("Failed to connect to SQLite", error=str(e))
            raise ConnectionError(f"Failed to connect to SQLite: {e}")

    async def disconnect(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
        self._connected = False
        self.log_info("Disconnected from SQLite")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test database connectivity."""
        try:
            if not self._connected:
                await self.connect()

            async with self._connection.execute("SELECT 1") as cursor:
                result = await cursor.fetchone()
                if result and result[0] == 1:
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

        # Get all tables
        async with self._connection.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ) as cursor:
            table_rows = await cursor.fetchall()

        for name, obj_type in table_rows:
            is_view = obj_type == "view"

            # Get columns
            columns = await self._get_table_columns(name)

            # Get primary key
            pk_columns = await self._get_primary_key(name)

            # Get foreign keys
            foreign_keys = await self._get_foreign_keys(name)

            # Get row count (for tables only)
            row_count = None
            if not is_view:
                row_count = await self._get_row_count(name)

            table_schema = TableSchema(
                name=name,
                schema_name="main",
                columns=columns,
                primary_key=pk_columns,
                foreign_keys=foreign_keys,
                row_count=row_count,
            )

            if is_view:
                views.append(table_schema)
            else:
                tables.append(table_schema)

        self._schema_cache = DatabaseSchema(
            database_name=self.database_path,
            connector_type=self.connector_type,
            tables=tables,
            views=views,
        )

        return self._schema_cache

    async def _get_table_columns(self, table_name: str) -> List[ColumnSchema]:
        """Get columns for a table."""
        columns = []

        # Get column info using PRAGMA
        async with self._connection.execute(f"PRAGMA table_info('{table_name}')") as cursor:
            rows = await cursor.fetchall()

        # Get foreign key info
        fk_map = {}
        async with self._connection.execute(f"PRAGMA foreign_key_list('{table_name}')") as cursor:
            fk_rows = await cursor.fetchall()
            for row in fk_rows:
                # row format: (id, seq, table, from, to, on_update, on_delete, match)
                fk_map[row[3]] = {"table": row[2], "column": row[4]}

        for row in rows:
            # row format: (cid, name, type, notnull, dflt_value, pk)
            cid, name, col_type, notnull, default_value, is_pk = row

            fk_info = fk_map.get(name)

            columns.append(ColumnSchema(
                name=name,
                data_type=col_type or "TEXT",
                is_nullable=not notnull,
                is_primary_key=is_pk == 1,
                is_foreign_key=fk_info is not None,
                foreign_key_table=fk_info["table"] if fk_info else None,
                foreign_key_column=fk_info["column"] if fk_info else None,
                default_value=str(default_value) if default_value is not None else None,
            ))

        return columns

    async def _get_primary_key(self, table_name: str) -> Optional[List[str]]:
        """Get primary key columns for a table."""
        pk_columns = []

        async with self._connection.execute(f"PRAGMA table_info('{table_name}')") as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            # row format: (cid, name, type, notnull, dflt_value, pk)
            if row[5] > 0:  # pk column is > 0 for primary key columns
                pk_columns.append(row[1])

        return pk_columns if pk_columns else None

    async def _get_foreign_keys(self, table_name: str) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table."""
        foreign_keys = []

        async with self._connection.execute(f"PRAGMA foreign_key_list('{table_name}')") as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            # row format: (id, seq, table, from, to, on_update, on_delete, match)
            foreign_keys.append({
                "column": row[3],
                "foreign_table": row[2],
                "foreign_column": row[4],
                "constraint_name": f"fk_{table_name}_{row[3]}",
            })

        return foreign_keys

    async def _get_row_count(self, table_name: str) -> Optional[int]:
        """Get row count for a table."""
        try:
            async with self._connection.execute(
                f"SELECT COUNT(*) FROM \"{table_name}\""
            ) as cursor:
                result = await cursor.fetchone()
                return result[0] if result else None
        except Exception:
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

            # Execute query
            if parameters:
                cursor = await self._connection.execute(query, tuple(parameters.values()))
            else:
                cursor = await self._connection.execute(query)

            # Get column names from cursor description
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            # Fetch all rows
            rows = await cursor.fetchall()
            await cursor.close()

            # Convert to list of lists
            data = [list(row) for row in rows]
            truncated = len(data) >= self.max_rows

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

        query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
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
            FROM "{table_name}"
            WHERE "{column_name}" IS NOT NULL
            LIMIT {limit}
        """

        result = await self.execute_query(query)
        if result.success and result.rows:
            return [row[0] for row in result.rows]
        return []

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for SQLite credentials."""
        return {
            "type": "object",
            "required": ["database_path"],
            "properties": {
                "database_path": {
                    "type": "string",
                    "description": "Path to SQLite database file (or :memory: for in-memory)",
                },
            },
        }
