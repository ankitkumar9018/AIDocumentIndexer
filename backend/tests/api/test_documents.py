"""
AIDocumentIndexer - Documents API Tests
=======================================

Integration tests for document management endpoints.
"""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock


# =============================================================================
# Document Listing Tests
# =============================================================================

class TestDocumentListing:
    """Tests for document listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_documents_authenticated(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test listing documents as authenticated user."""
        response = await async_client.get(
            "/api/documents",
            headers=auth_headers,
        )

        # Should get 200 with empty list (no documents in test db)
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_documents_unauthenticated(self, async_client: AsyncClient):
        """Test listing documents without authentication."""
        response = await async_client.get("/api/documents")

        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_list_documents_pagination(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test document listing with pagination parameters."""
        response = await async_client.get(
            "/api/documents",
            headers=auth_headers,
            params={"page": 1, "page_size": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10


# =============================================================================
# Document Search Tests
# =============================================================================

class TestDocumentSearch:
    """Tests for document search endpoint."""

    @pytest.mark.asyncio
    async def test_search_documents(
        self, async_client: AsyncClient, auth_headers: dict, mock_vector_store
    ):
        """Test searching documents."""
        response = await async_client.post(
            "/api/documents/search",
            headers=auth_headers,
            json={"query": "test query", "limit": 10},
        )

        # Should return empty results with mocked vector store
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_results" in data

    @pytest.mark.asyncio
    async def test_search_documents_with_filters(
        self, async_client: AsyncClient, auth_headers: dict, mock_vector_store
    ):
        """Test searching documents with filters."""
        response = await async_client.post(
            "/api/documents/search",
            headers=auth_headers,
            json={
                "query": "test query",
                "collection": "test_collection",
                "file_types": ["pdf"],
                "limit": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_search_documents_empty_query(
        self, async_client: AsyncClient, auth_headers: dict, mock_vector_store
    ):
        """Test searching with empty query."""
        response = await async_client.post(
            "/api/documents/search",
            headers=auth_headers,
            json={"query": "", "limit": 10},
        )

        # Should return results (empty query is valid)
        assert response.status_code == 200


# =============================================================================
# Document CRUD Tests
# =============================================================================

class TestDocumentCRUD:
    """Tests for document CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test getting a non-existent document."""
        response = await async_client.get(
            "/api/documents/550e8400-e29b-41d4-a716-446655440000",
            headers=auth_headers,
        )

        # Document doesn't exist, should be 404
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_document_not_found(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test updating a non-existent document."""
        response = await async_client.patch(
            "/api/documents/550e8400-e29b-41d4-a716-446655440000",
            headers=auth_headers,
            json={"name": "Updated Name"},
        )

        # Document doesn't exist, should be 404
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_document_not_found(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test deleting a non-existent document."""
        response = await async_client.delete(
            "/api/documents/550e8400-e29b-41d4-a716-446655440000",
            headers=auth_headers,
        )

        # Document doesn't exist, should be 404
        assert response.status_code == 404


# =============================================================================
# Document Chunks Tests
# =============================================================================

class TestDocumentChunks:
    """Tests for document chunks endpoint."""

    @pytest.mark.asyncio
    async def test_get_document_chunks_not_found(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test getting chunks for non-existent document."""
        response = await async_client.get(
            "/api/documents/550e8400-e29b-41d4-a716-446655440000/chunks",
            headers=auth_headers,
        )

        # Document doesn't exist, user doesn't have access
        assert response.status_code == 403


# =============================================================================
# Collections Tests
# =============================================================================

class TestCollections:
    """Tests for collections endpoint."""

    @pytest.mark.asyncio
    async def test_list_collections(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test listing collections."""
        response = await async_client.get(
            "/api/documents/collections/list",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert isinstance(data["collections"], list)


# =============================================================================
# Document Reprocessing Tests
# =============================================================================

class TestDocumentReprocessing:
    """Tests for document reprocessing endpoint."""

    @pytest.mark.asyncio
    async def test_reprocess_document_not_found(
        self, async_client: AsyncClient, auth_headers: dict
    ):
        """Test reprocessing a non-existent document."""
        with patch("backend.api.routes.documents.get_permission_service") as mock_perm:
            mock_perm_service = AsyncMock()
            mock_perm_service.check_document_access.return_value = False
            mock_perm.return_value = mock_perm_service

            response = await async_client.post(
                "/api/documents/550e8400-e29b-41d4-a716-446655440000/reprocess",
                headers=auth_headers,
            )

            # Either no permission (403) or not found (404)
            assert response.status_code in [403, 404]
