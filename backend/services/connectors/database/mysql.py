"""
AIDocumentIndexer - MySQL Database Connector
=============================================

MySQL connector for natural language querying using aiomysql.

Features:
- Async connection pooling
- Schema introspection with foreign key detection
- Read-only query execution with statement timeout
- Sample data retrieval for few-shot prompting
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import structlog

try:
    import aiomysql
    HAS_AIOMYSQL = True
except ImportError:
    aiomysql = None
    HAS_AIOMYSQL = False

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
    Sanitize a SQL identifier (table/column name) to prevent injection.

    Only allows alphanumeric characters and underscores.
    Raises ValueError for invalid identifiers.
    """
    import re
    if not name or not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


class MySQLConnector(BaseDatabaseConnector):
    """
    MySQL database connector using aiomysql.

    Supports:
    - Connection pooling for efficient query execution
    - Full schema introspection including foreign keys
    - Read-only query execution with timeout
    - Sample data retrieval
    """

    connector_type = DatabaseConnectorType.MYSQL
    display_name = "MySQL"
    description = "Connect to MySQL databases for natural language querying"

    def __init__(
        self,
        host: str,
        port: int = 3306,
        database: str = "",
        username: str = "root",
        password: str = "",
        ssl_mode: Optional[str] = None,
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
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool: Optional[aiomysql.Pool] = None

    async def connect(self) -> bool:
        """Establish connection pool to MySQL."""
        if not HAS_AIOMYSQL:
            raise ImportError(
                "aiomysql is required for MySQL connections. "
                "Install with: pip install aiomysql"
            )

        try:
            # Create connection pool
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                db=self.database,
                user=self.username,
                password=self.password,
                minsize=self.pool_min_size,
                maxsize=self.pool_max_size,
                connect_timeout=self.query_timeout_seconds,
                autocommit=True,
            )

            self._connected = True
            self.log_info("Connected to MySQL", host=self.host, database=self.database)
            return True

        except Exception as e:
            self.log_error("Failed to connect to MySQL", error=str(e))
            raise ConnectionError(f"Failed to connect to MySQL: {e}")

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
        self._connected = False
        self.log_info("Disconnected from MySQL")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test database connectivity."""
        try:
            if not self._connected:
                await self.connect()

            async with self._pool.acquire() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    result = await cur.fetchone()
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

        async with self._pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # Get tables and views
                await cur.execute("""
                    SELECT TABLE_NAME, TABLE_TYPE
                    FROM INFORMATION_SCHEMA.TABLES
                    WHERE TABLE_SCHEMA = %s
                    ORDER BY TABLE_NAME
                """, (self.database,))
                table_rows = await cur.fetchall()

                for row in table_rows:
                    table_name = row["TABLE_NAME"]
                    is_view = row["TABLE_TYPE"] == "VIEW"

                    # Get columns
                    columns = await self._get_table_columns(cur, table_name)

                    # Get primary key
                    pk_columns = await self._get_primary_key(cur, table_name)

                    # Get foreign keys
                    foreign_keys = await self._get_foreign_keys(cur, table_name)

                    # Get row count estimate (for tables only)
                    row_count = None
                    if not is_view:
                        row_count = await self._get_row_count_estimate(cur, table_name)

                    table_schema = TableSchema(
                        name=table_name,
                        schema_name=self.database,
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
            database_name=self.database,
            connector_type=self.connector_type,
            tables=tables,
            views=views,
        )

        return self._schema_cache

    async def _get_table_columns(
        self,
        cur,
        table_name: str,
    ) -> List[ColumnSchema]:
        """Get columns for a table."""
        columns = []

        await cur.execute("""
            SELECT
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                c.COLUMN_DEFAULT,
                c.CHARACTER_MAXIMUM_LENGTH,
                c.NUMERIC_PRECISION,
                c.NUMERIC_SCALE,
                c.COLUMN_KEY
            FROM INFORMATION_SCHEMA.COLUMNS c
            WHERE c.TABLE_SCHEMA = %s AND c.TABLE_NAME = %s
            ORDER BY c.ORDINAL_POSITION
        """, (self.database, table_name))
        rows = await cur.fetchall()

        # Get foreign keys separately
        await cur.execute("""
            SELECT
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
              AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (self.database, table_name))
        fk_rows = await cur.fetchall()
        fk_map = {row["COLUMN_NAME"]: row for row in fk_rows}

        for row in rows:
            # Build data type string with precision info
            data_type = row["DATA_TYPE"]
            if row["CHARACTER_MAXIMUM_LENGTH"]:
                data_type += f"({row['CHARACTER_MAXIMUM_LENGTH']})"
            elif row["NUMERIC_PRECISION"] and row["DATA_TYPE"] in ("decimal", "numeric"):
                scale = row["NUMERIC_SCALE"] or 0
                data_type += f"({row['NUMERIC_PRECISION']},{scale})"

            fk_info = fk_map.get(row["COLUMN_NAME"])

            columns.append(ColumnSchema(
                name=row["COLUMN_NAME"],
                data_type=data_type,
                is_nullable=row["IS_NULLABLE"] == "YES",
                is_primary_key=row["COLUMN_KEY"] == "PRI",
                is_foreign_key=fk_info is not None,
                foreign_key_table=fk_info["REFERENCED_TABLE_NAME"] if fk_info else None,
                foreign_key_column=fk_info["REFERENCED_COLUMN_NAME"] if fk_info else None,
                default_value=row["COLUMN_DEFAULT"],
            ))

        return columns

    async def _get_primary_key(
        self,
        cur,
        table_name: str,
    ) -> Optional[List[str]]:
        """Get primary key columns for a table."""
        await cur.execute("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
              AND CONSTRAINT_NAME = 'PRIMARY'
            ORDER BY ORDINAL_POSITION
        """, (self.database, table_name))
        rows = await cur.fetchall()

        if rows:
            return [row["COLUMN_NAME"] for row in rows]
        return None

    async def _get_foreign_keys(
        self,
        cur,
        table_name: str,
    ) -> List[Dict[str, Any]]:
        """Get foreign key relationships for a table."""
        await cur.execute("""
            SELECT
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME,
                CONSTRAINT_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME = %s
              AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (self.database, table_name))
        rows = await cur.fetchall()

        return [
            {
                "column": row["COLUMN_NAME"],
                "foreign_table": row["REFERENCED_TABLE_NAME"],
                "foreign_column": row["REFERENCED_COLUMN_NAME"],
                "constraint_name": row["CONSTRAINT_NAME"],
            }
            for row in rows
        ]

    async def _get_row_count_estimate(
        self,
        cur,
        table_name: str,
    ) -> Optional[int]:
        """Get estimated row count from table statistics."""
        try:
            await cur.execute("""
                SELECT TABLE_ROWS
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (self.database, table_name))
            result = await cur.fetchone()
            if result and result["TABLE_ROWS"]:
                return int(result["TABLE_ROWS"])
            return None
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
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    # Set query timeout
                    await cur.execute(f"SET max_execution_time = {self.query_timeout_seconds * 1000}")

                    # Execute in read-only mode by starting a read-only transaction
                    await cur.execute("START TRANSACTION READ ONLY")

                    try:
                        if parameters:
                            await cur.execute(query, tuple(parameters.values()))
                        else:
                            await cur.execute(query)

                        rows = await cur.fetchall()
                        await cur.execute("COMMIT")
                    except Exception:
                        await cur.execute("ROLLBACK")
                        raise

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

        except Exception as e:
            error_msg = str(e)
            if "max_execution_time" in error_msg.lower() or "query execution was interrupted" in error_msg.lower():
                error_msg = f"Query timed out after {self.query_timeout_seconds} seconds"

            self.log_error("Query execution failed", error=error_msg, query=query)
            return QueryResult(
                success=False,
                error=error_msg,
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

        # Sanitize identifier to prevent SQL injection
        safe_table = _sanitize_identifier(table_name)
        query = f"SELECT * FROM `{safe_table}` LIMIT {int(limit)}"
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
            SELECT DISTINCT `{column_name}`
            FROM `{table_name}`
            WHERE `{column_name}` IS NOT NULL
            LIMIT {limit}
        """

        result = await self.execute_query(query)
        if result.success and result.rows:
            return [row[0] for row in result.rows]
        return []

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for MySQL credentials."""
        base_schema = super().get_credentials_schema()
        base_schema["properties"]["port"]["default"] = 3306
        return base_schema
