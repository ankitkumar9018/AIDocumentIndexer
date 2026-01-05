/**
 * AIDocumentIndexer - Saved Searches API Integration Tests
 * =========================================================
 *
 * Tests for saved search CRUD operations via the API client.
 */

import { api } from '@/lib/api/client';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('Saved Searches API', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    // Set the auth token on the API client directly
    api.setToken('mock-auth-token');
  });

  afterEach(() => {
    api.clearToken();
  });

  describe('listSavedSearches', () => {
    it('fetches all saved searches for the user', async () => {
      const mockSearches = {
        searches: [
          {
            name: 'Marketing Docs',
            query: 'marketing campaign',
            collection: 'Marketing',
            search_mode: 'hybrid',
            created_at: '2026-01-01T00:00:00Z',
          },
          {
            name: 'Recent PDFs',
            query: '',
            file_types: ['pdf'],
            search_mode: 'keyword',
            created_at: '2026-01-02T00:00:00Z',
          },
        ],
        count: 2,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearches),
      });

      const result = await api.listSavedSearches();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/preferences/searches'),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer mock-auth-token',
          }),
        })
      );
      expect(result).toEqual(mockSearches);
      expect(result.count).toBe(2);
    });

    it('returns empty list when no saved searches exist', async () => {
      const mockSearches = {
        searches: [],
        count: 0,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearches),
      });

      const result = await api.listSavedSearches();

      expect(result.searches).toHaveLength(0);
      expect(result.count).toBe(0);
    });
  });

  describe('getSavedSearch', () => {
    it('fetches a specific saved search by name', async () => {
      const mockSearch = {
        name: 'Marketing Docs',
        query: 'marketing campaign',
        collection: 'Marketing',
        folder_id: null,
        include_subfolders: true,
        date_from: null,
        date_to: null,
        file_types: null,
        search_mode: 'hybrid',
        created_at: '2026-01-01T00:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSearch),
      });

      const result = await api.getSavedSearch('Marketing Docs');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/preferences/searches/Marketing%20Docs'),
        expect.any(Object)
      );
      expect(result).toEqual(mockSearch);
    });

    it('throws error when search not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'NOT_FOUND',
          message: "Saved search 'NonExistent' not found"
        }),
      });

      await expect(api.getSavedSearch('NonExistent')).rejects.toThrow();
    });
  });

  describe('saveSearch', () => {
    it('creates a new saved search', async () => {
      const newSearch = {
        name: 'New Search',
        query: 'test query',
        collection: 'TestCollection',
        folder_id: null,
        include_subfolders: true,
        date_from: null,
        date_to: null,
        file_types: ['pdf', 'docx'],
        search_mode: 'hybrid',
      };

      const savedSearch = {
        ...newSearch,
        created_at: '2026-01-02T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(savedSearch),
      });

      const result = await api.saveSearch(newSearch);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/preferences/searches'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify(newSearch),
        })
      );
      expect(result.name).toBe('New Search');
      expect(result.created_at).toBeDefined();
    });

    it('updates an existing saved search with same name', async () => {
      const updatedSearch = {
        name: 'Existing Search',
        query: 'updated query',
        collection: 'UpdatedCollection',
        search_mode: 'vector',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...updatedSearch,
          created_at: '2026-01-02T12:00:00Z',
        }),
      });

      const result = await api.saveSearch(updatedSearch);

      expect(result.query).toBe('updated query');
      expect(result.collection).toBe('UpdatedCollection');
    });

    it('throws error when max searches limit reached', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'BAD_REQUEST',
          message: 'Maximum 20 saved searches allowed. Delete some to add more.'
        }),
      });

      await expect(api.saveSearch({ name: 'Too Many', query: '' })).rejects.toThrow();
    });
  });

  describe('deleteSavedSearch', () => {
    it('deletes a saved search by name', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ message: "Deleted saved search 'Old Search'" }),
      });

      const result = await api.deleteSavedSearch('Old Search');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/preferences/searches/Old%20Search'),
        expect.objectContaining({
          method: 'DELETE',
        })
      );
      expect(result.message).toContain('Deleted');
    });

    it('throws error when search to delete not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({
          error: 'NOT_FOUND',
          message: "Saved search 'NonExistent' not found"
        }),
      });

      await expect(api.deleteSavedSearch('NonExistent')).rejects.toThrow();
    });
  });

  describe('search filter combinations', () => {
    it('saves search with folder filter', async () => {
      const searchWithFolder = {
        name: 'Folder Search',
        query: 'documents',
        folder_id: 'folder-123',
        include_subfolders: true,
        search_mode: 'hybrid',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...searchWithFolder,
          created_at: '2026-01-02T12:00:00Z',
        }),
      });

      const result = await api.saveSearch(searchWithFolder);

      expect(result.folder_id).toBe('folder-123');
      expect(result.include_subfolders).toBe(true);
    });

    it('saves search with date range filter', async () => {
      const searchWithDates = {
        name: 'Date Range Search',
        query: 'reports',
        date_from: '2025-01-01',
        date_to: '2025-12-31',
        search_mode: 'keyword',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...searchWithDates,
          created_at: '2026-01-02T12:00:00Z',
        }),
      });

      const result = await api.saveSearch(searchWithDates);

      expect(result.date_from).toBe('2025-01-01');
      expect(result.date_to).toBe('2025-12-31');
    });

    it('saves search with multiple file types', async () => {
      const searchWithTypes = {
        name: 'Multi-type Search',
        query: '',
        file_types: ['pdf', 'docx', 'xlsx'],
        search_mode: 'hybrid',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          ...searchWithTypes,
          created_at: '2026-01-02T12:00:00Z',
        }),
      });

      const result = await api.saveSearch(searchWithTypes);

      expect(result.file_types).toEqual(['pdf', 'docx', 'xlsx']);
    });
  });

  describe('authentication handling', () => {
    it('includes auth token in all requests', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ searches: [], count: 0 }),
      });

      await api.listSavedSearches();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer mock-auth-token',
          }),
        })
      );
    });

    it('throws error on unauthorized request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ error: 'UNAUTHORIZED', message: 'Invalid token' }),
      });

      await expect(api.listSavedSearches()).rejects.toThrow();
    });
  });
});
