/**
 * AIDocumentIndexer - Folder API Integration Tests
 * =================================================
 *
 * Tests for folder CRUD operations via the API client.
 */

import { api } from '@/lib/api/client';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('Folder API', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    // Set the auth token on the API client directly
    api.setToken('mock-auth-token');
  });

  afterEach(() => {
    api.clearToken();
  });

  describe('listFolders', () => {
    it('fetches root folders when no parent_id provided', async () => {
      const mockFolders = [
        { id: 'folder-1', name: 'Documents', path: '/Documents/', depth: 0 },
        { id: 'folder-2', name: 'Images', path: '/Images/', depth: 0 },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockFolders),
      });

      const result = await api.listFolders();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders'),
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer mock-auth-token',
          }),
        })
      );
      expect(result).toEqual(mockFolders);
    });

    it('fetches child folders when parent_id provided', async () => {
      const mockFolders = [
        { id: 'folder-3', name: 'Subfolder', path: '/Documents/Subfolder/', depth: 1 },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockFolders),
      });

      const result = await api.listFolders({ parent_id: 'folder-1' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('parent_id=folder-1'),
        expect.any(Object)
      );
      expect(result).toEqual(mockFolders);
    });
  });

  describe('getFolderTree', () => {
    it('fetches the complete folder tree', async () => {
      const mockTree = [
        {
          id: 'folder-1',
          name: 'Documents',
          path: '/Documents/',
          depth: 0,
          children: [
            { id: 'folder-3', name: 'Subfolder', path: '/Documents/Subfolder/', depth: 1, children: [] },
          ],
        },
        { id: 'folder-2', name: 'Images', path: '/Images/', depth: 0, children: [] },
      ];

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockTree),
      });

      const result = await api.getFolderTree();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders/tree'),
        expect.any(Object)
      );
      expect(result).toEqual(mockTree);
    });
  });

  describe('createFolder', () => {
    it('creates a new root folder', async () => {
      const newFolder = {
        id: 'new-folder-1',
        name: 'New Folder',
        path: '/New Folder/',
        depth: 0,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(newFolder),
      });

      const result = await api.createFolder({
        name: 'New Folder',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ name: 'New Folder' }),
        })
      );
      expect(result).toEqual(newFolder);
    });

    it('creates a subfolder with parent_id', async () => {
      const newFolder = {
        id: 'new-folder-2',
        name: 'Subfolder',
        path: '/Documents/Subfolder/',
        depth: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(newFolder),
      });

      const result = await api.createFolder({
        name: 'Subfolder',
        parent_folder_id: 'folder-1',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            name: 'Subfolder',
            parent_folder_id: 'folder-1',
          }),
        })
      );
      expect(result).toEqual(newFolder);
    });
  });

  describe('updateFolder', () => {
    it('updates folder name', async () => {
      const updatedFolder = {
        id: 'folder-1',
        name: 'Renamed Folder',
        path: '/Renamed Folder/',
        depth: 0,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(updatedFolder),
      });

      const result = await api.updateFolder('folder-1', { name: 'Renamed Folder' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders/folder-1'),
        expect.objectContaining({
          method: 'PATCH',
          body: JSON.stringify({ name: 'Renamed Folder' }),
        })
      );
      expect(result).toEqual(updatedFolder);
    });
  });

  describe('deleteFolder', () => {
    it('deletes an empty folder', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ message: 'Folder deleted successfully' }),
      });

      const result = await api.deleteFolder('folder-1');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders/folder-1'),
        expect.objectContaining({
          method: 'DELETE',
        })
      );
      expect(result).toEqual({ message: 'Folder deleted successfully' });
    });

    it('deletes folder recursively when specified', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ message: 'Folder and contents deleted' }),
      });

      const result = await api.deleteFolder('folder-1', true);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('recursive=true'),
        expect.objectContaining({
          method: 'DELETE',
        })
      );
      expect(result).toEqual({ message: 'Folder and contents deleted' });
    });
  });

  describe('moveFolder', () => {
    it('moves folder to new parent', async () => {
      const movedFolder = {
        id: 'folder-3',
        name: 'Subfolder',
        path: '/Images/Subfolder/',
        depth: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(movedFolder),
      });

      const result = await api.moveFolder('folder-3', 'folder-2');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders/folder-3/move'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ new_parent_id: 'folder-2' }),
        })
      );
      expect(result).toEqual(movedFolder);
    });

    it('moves folder to root when parent is null', async () => {
      const movedFolder = {
        id: 'folder-3',
        name: 'Subfolder',
        path: '/Subfolder/',
        depth: 0,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(movedFolder),
      });

      const result = await api.moveFolder('folder-3', null);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/folders/folder-3/move'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ new_parent_id: null }),
        })
      );
      expect(result).toEqual(movedFolder);
    });
  });

  describe('error handling', () => {
    it('throws error on unauthorized request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ error: 'UNAUTHORIZED', message: 'Invalid token' }),
      });

      await expect(api.listFolders()).rejects.toThrow();
    });

    it('throws error when folder not found', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'NOT_FOUND', message: 'Folder not found' }),
      });

      await expect(api.getFolder('non-existent')).rejects.toThrow();
    });

    it('throws error on server error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ error: 'INTERNAL_ERROR', message: 'Server error' }),
      });

      await expect(api.createFolder({ name: 'Test' })).rejects.toThrow();
    });
  });
});
