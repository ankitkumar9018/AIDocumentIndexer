"""
AIDocumentIndexer - MongoDB Database Connector
===============================================

Connector for MongoDB databases using motor async driver.
Provides natural language querying of MongoDB collections.

Note: MongoDB is a document database, so we translate natural language
to MongoDB aggregation pipelines instead of SQL.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog

from .base import (
    BaseDatabaseConnector,
    ColumnSchema,
    DatabaseConnectorType,
    DatabaseSchema,
    QueryResult,
    QueryValidationResult,
    TableSchema,
)

logger = structlog.get_logger(__name__)

# MongoDB-specific dangerous operations
DANGEROUS_MONGO_OPERATIONS = {
    "$where",  # JavaScript execution
    "$function",  # User-defined functions
    "$accumulator",  # Custom accumulator with JS
    "mapReduce",  # Map-reduce operations
    "$out",  # Write to collection
    "$merge",  # Merge into collection
}


class MongoDBConnector(BaseDatabaseConnector):
    """
    MongoDB database connector using motor async driver.

    Translates natural language queries into MongoDB aggregation pipelines.
    Only read operations (find, aggregate) are allowed.
    """

    connector_type = DatabaseConnectorType.MONGODB
    display_name = "MongoDB"
    description = "Connect to MongoDB databases for natural language querying"

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
        auth_source: str = "admin",
        replica_set: Optional[str] = None,
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
        self.auth_source = auth_source
        self.replica_set = replica_set
        self._client = None
        self._db = None

    def _build_connection_uri(self) -> str:
        """Build MongoDB connection URI."""
        # Build URI with authentication
        if self.username and self.password:
            # URL encode special characters in password
            from urllib.parse import quote_plus
            password_encoded = quote_plus(self.password)
            uri = f"mongodb://{self.username}:{password_encoded}@{self.host}:{self.port}"
        else:
            uri = f"mongodb://{self.host}:{self.port}"

        # Add options
        options = []
        options.append(f"authSource={self.auth_source}")

        if self.replica_set:
            options.append(f"replicaSet={self.replica_set}")

        if self.ssl_mode and self.ssl_mode != "disable":
            options.append("tls=true")
            if self.ssl_mode == "verify-full":
                options.append("tlsAllowInvalidCertificates=false")
            elif self.ssl_mode in ("allow", "prefer"):
                options.append("tlsAllowInvalidCertificates=true")

        if options:
            uri += "?" + "&".join(options)

        return uri

    async def connect(self) -> bool:
        """Establish connection to MongoDB."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            uri = self._build_connection_uri()
            self._client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=self.query_timeout_seconds * 1000,
                connectTimeoutMS=10000,
            )
            self._db = self._client[self.database]

            # Test connection
            await self._client.admin.command('ping')

            self._connected = True
            self.log_info("Connected to MongoDB")
            return True

        except ImportError:
            self.log_error("motor package not installed. Run: pip install motor")
            raise ConnectionError("motor package required for MongoDB connections")
        except Exception as e:
            self.log_error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionError(f"MongoDB connection failed: {e}")

    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._connected = False
            self.log_info("Disconnected from MongoDB")

    async def test_connection(self) -> Tuple[bool, Optional[str]]:
        """Test MongoDB connectivity."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            uri = self._build_connection_uri()
            client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )

            # Ping the server
            await client.admin.command('ping')

            # Get database stats
            db = client[self.database]
            stats = await db.command('dbStats')

            client.close()

            return True, None

        except ImportError:
            return False, "motor package not installed"
        except Exception as e:
            return False, str(e)

    async def get_schema(self, refresh: bool = False) -> DatabaseSchema:
        """
        Get MongoDB database schema.

        For MongoDB, we introspect collections and sample documents
        to infer the schema structure.
        """
        if self._schema_cache and not refresh:
            return self._schema_cache

        if not self._connected:
            await self.connect()

        tables = []

        try:
            # Get collection names
            collection_names = await self._db.list_collection_names()

            for collection_name in collection_names:
                # Skip system collections
                if collection_name.startswith('system.'):
                    continue

                collection = self._db[collection_name]

                # Get collection stats
                try:
                    stats = await self._db.command('collStats', collection_name)
                    row_count = stats.get('count', 0)
                except Exception:
                    row_count = await collection.estimated_document_count()

                # Sample documents to infer schema
                columns = await self._infer_collection_schema(collection)

                table = TableSchema(
                    name=collection_name,
                    columns=columns,
                    row_count=row_count,
                    description=f"MongoDB collection: {collection_name}",
                )
                tables.append(table)

            self._schema_cache = DatabaseSchema(
                database_name=self.database,
                connector_type=self.connector_type,
                tables=tables,
                last_updated=datetime.utcnow(),
            )

            self.log_info(f"Retrieved schema with {len(tables)} collections")
            return self._schema_cache

        except Exception as e:
            self.log_error(f"Failed to get schema: {e}")
            raise

    async def _infer_collection_schema(
        self,
        collection,
        sample_size: int = 100,
    ) -> List[ColumnSchema]:
        """
        Infer schema from sample documents in a collection.

        MongoDB is schemaless, so we sample documents and aggregate
        the field types found.
        """
        columns = []
        field_types: Dict[str, Dict[str, int]] = {}
        field_samples: Dict[str, List[Any]] = {}

        try:
            # Sample documents
            cursor = collection.find().limit(sample_size)
            async for doc in cursor:
                self._extract_fields(doc, "", field_types, field_samples)

            # Build column schemas from aggregated types
            for field_path, type_counts in field_types.items():
                # Get most common type
                most_common_type = max(type_counts, key=type_counts.get)

                # Determine if nullable (if we saw null or missing)
                is_nullable = "null" in type_counts or type_counts.get(most_common_type, 0) < sample_size

                column = ColumnSchema(
                    name=field_path,
                    data_type=most_common_type,
                    is_nullable=is_nullable,
                    is_primary_key=field_path == "_id",
                    sample_values=field_samples.get(field_path, [])[:3],
                )
                columns.append(column)

        except Exception as e:
            logger.warning(f"Error inferring schema: {e}")

        return columns

    def _extract_fields(
        self,
        doc: Dict[str, Any],
        prefix: str,
        field_types: Dict[str, Dict[str, int]],
        field_samples: Dict[str, List[Any]],
        max_depth: int = 3,
    ) -> None:
        """Recursively extract fields from a document."""
        if max_depth <= 0:
            return

        for key, value in doc.items():
            field_path = f"{prefix}.{key}" if prefix else key

            # Get type name
            type_name = self._get_mongo_type_name(value)

            # Track type
            if field_path not in field_types:
                field_types[field_path] = {}
                field_samples[field_path] = []

            field_types[field_path][type_name] = field_types[field_path].get(type_name, 0) + 1

            # Store sample (avoiding large values)
            if len(field_samples[field_path]) < 5:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    field_samples[field_path].append(value)

            # Recurse into nested documents (but not arrays)
            if isinstance(value, dict) and not key.startswith('$'):
                self._extract_fields(value, field_path, field_types, field_samples, max_depth - 1)

    def _get_mongo_type_name(self, value: Any) -> str:
        """Get MongoDB type name for a value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "double"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        elif isinstance(value, datetime):
            return "date"
        elif hasattr(value, '__class__') and value.__class__.__name__ == 'ObjectId':
            return "objectId"
        else:
            return type(value).__name__

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute a MongoDB query.

        The query should be a JSON string representing either:
        1. A find filter: {"field": "value"}
        2. An aggregation pipeline: [{"$match": {...}}, {"$group": {...}}]

        The query string format:
        - "collection_name|find|{filter}" for find queries
        - "collection_name|aggregate|[pipeline]" for aggregation
        """
        if not self._connected:
            await self.connect()

        start_time = time.time()

        try:
            import json

            # Parse query format: collection|operation|query_json
            parts = query.split("|", 2)
            if len(parts) != 3:
                return QueryResult(
                    success=False,
                    error="Query format: collection_name|operation|query_json",
                    query=query,
                )

            collection_name, operation, query_json = parts
            collection = self._db[collection_name]

            # Parse query JSON
            try:
                query_data = json.loads(query_json)
            except json.JSONDecodeError as e:
                return QueryResult(
                    success=False,
                    error=f"Invalid JSON in query: {e}",
                    query=query,
                )

            # Validate query
            validation = self.validate_mongo_query(operation, query_data)
            if not validation.is_valid:
                return QueryResult(
                    success=False,
                    error=f"Query validation failed: {'; '.join(validation.errors)}",
                    query=query,
                )

            rows = []
            columns = []

            if operation.lower() == "find":
                # Execute find query
                filter_doc = query_data if isinstance(query_data, dict) else {}
                cursor = collection.find(filter_doc).limit(self.max_rows)

                async for doc in cursor:
                    # Convert ObjectId to string
                    doc = self._serialize_document(doc)

                    if not columns:
                        columns = list(doc.keys())

                    rows.append([doc.get(col) for col in columns])

            elif operation.lower() == "aggregate":
                # Execute aggregation pipeline
                pipeline = query_data if isinstance(query_data, list) else [query_data]

                # Add $limit if not present
                has_limit = any("$limit" in stage for stage in pipeline)
                if not has_limit:
                    pipeline.append({"$limit": self.max_rows})

                cursor = collection.aggregate(pipeline)

                async for doc in cursor:
                    doc = self._serialize_document(doc)

                    if not columns:
                        columns = list(doc.keys())

                    rows.append([doc.get(col) for col in columns])

            else:
                return QueryResult(
                    success=False,
                    error=f"Unsupported operation: {operation}. Use 'find' or 'aggregate'",
                    query=query,
                )

            execution_time = (time.time() - start_time) * 1000

            return QueryResult(
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=execution_time,
                query=query,
                truncated=len(rows) >= self.max_rows,
            )

        except Exception as e:
            self.log_error(f"Query execution failed: {e}")
            return QueryResult(
                success=False,
                error=str(e),
                query=query,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _serialize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize MongoDB document for JSON output."""
        result = {}
        for key, value in doc.items():
            if hasattr(value, '__class__') and value.__class__.__name__ == 'ObjectId':
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_document(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_document(v) if isinstance(v, dict) else str(v) if hasattr(v, '__class__') and v.__class__.__name__ == 'ObjectId' else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def validate_mongo_query(
        self,
        operation: str,
        query_data: Any,
    ) -> QueryValidationResult:
        """
        Validate MongoDB query for safety.

        Blocks dangerous operations like $where, $function, $out, $merge.
        """
        result = QueryValidationResult(
            is_valid=True,
            is_read_only=True,
            errors=[],
            warnings=[],
            tables_referenced=[],
        )

        # Only allow find and aggregate
        if operation.lower() not in ("find", "aggregate"):
            result.is_valid = False
            result.is_read_only = False
            result.errors.append(f"Operation '{operation}' not allowed. Use 'find' or 'aggregate'")
            return result

        result.query_type = operation.upper()

        # Check for dangerous operators
        def check_dangerous(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in DANGEROUS_MONGO_OPERATIONS:
                        result.is_valid = False
                        result.is_read_only = False
                        result.errors.append(f"Dangerous operator not allowed: {key}")
                    check_dangerous(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_dangerous(item, f"{path}[{i}]")

        check_dangerous(query_data)

        return result

    def validate_query(self, query: str) -> QueryValidationResult:
        """
        Validate a MongoDB query string.

        Expected format: collection_name|operation|query_json
        """
        try:
            import json

            parts = query.split("|", 2)
            if len(parts) != 3:
                return QueryValidationResult(
                    is_valid=False,
                    is_read_only=False,
                    errors=["Query format must be: collection_name|operation|query_json"],
                )

            collection_name, operation, query_json = parts

            try:
                query_data = json.loads(query_json)
            except json.JSONDecodeError as e:
                return QueryValidationResult(
                    is_valid=False,
                    is_read_only=False,
                    errors=[f"Invalid JSON: {e}"],
                )

            result = self.validate_mongo_query(operation, query_data)
            result.tables_referenced = [collection_name]
            return result

        except Exception as e:
            return QueryValidationResult(
                is_valid=False,
                is_read_only=False,
                errors=[str(e)],
            )

    async def get_sample_data(
        self,
        table_name: str,
        limit: int = 5,
    ) -> QueryResult:
        """Get sample documents from a collection."""
        if not self._connected:
            await self.connect()

        start_time = time.time()

        try:
            collection = self._db[table_name]
            cursor = collection.find().limit(limit)

            rows = []
            columns = []

            async for doc in cursor:
                doc = self._serialize_document(doc)

                if not columns:
                    columns = list(doc.keys())

                rows.append([doc.get(col) for col in columns])

            return QueryResult(
                success=True,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=(time.time() - start_time) * 1000,
                query=f"{table_name}|find|{{}}",
            )

        except Exception as e:
            return QueryResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    @classmethod
    def get_credentials_schema(cls) -> Dict[str, Any]:
        """Get JSON schema for MongoDB credentials."""
        schema = super().get_credentials_schema()
        schema["properties"]["auth_source"] = {
            "type": "string",
            "default": "admin",
            "description": "Authentication database",
        }
        schema["properties"]["replica_set"] = {
            "type": "string",
            "description": "Replica set name (optional)",
        }
        schema["properties"]["port"]["default"] = 27017
        return schema
