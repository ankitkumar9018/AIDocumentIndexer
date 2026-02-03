import { useState, useEffect } from 'react';
import {
  FileText,
  Folder,
  FolderPlus,
  FilePlus,
  Trash2,
  Loader2,
  RefreshCw,
  Eye,
} from 'lucide-react';
import { useAppStore, type Document, type WatchedFolder } from '../lib/store';
import {
  getDocuments,
  indexFile,
  indexFolder,
  deleteDocument,
  selectFile,
  selectFolder,
  addWatchFolder,
  removeWatchFolder,
  getWatchFolders,
  type IndexedDocument,
} from '../lib/tauri';
import { cn, formatFileSize, formatDate } from '../lib/utils';

export function DocumentsPage() {
  const {
    documents,
    setDocuments,
    removeDocument: removeDocFromStore,
    watchedFolders,
    setWatchedFolders,
    addWatchedFolder,
    removeWatchedFolder,
  } = useAppStore();
  const [isLoading, setIsLoading] = useState(true);
  const [isIndexing, setIsIndexing] = useState(false);
  const [activeTab, setActiveTab] = useState<'documents' | 'folders'>(
    'documents'
  );

  useEffect(() => {
    loadDocuments();
    loadWatchFolders();
  }, []);

  const loadDocuments = async () => {
    setIsLoading(true);
    try {
      const docs = await getDocuments();
      setDocuments(
        docs.map((doc) => ({
          ...doc,
          status: 'indexed' as const,
        }))
      );
    } catch (error) {
      console.error('Failed to load documents:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadWatchFolders = async () => {
    try {
      const folders = await getWatchFolders();
      setWatchedFolders(
        folders.map((path) => ({
          path,
          enabled: true,
          file_count: 0,
        }))
      );
    } catch (error) {
      console.error('Failed to load watch folders:', error);
    }
  };

  const handleAddFile = async () => {
    const path = await selectFile();
    if (!path) return;

    setIsIndexing(true);
    try {
      const doc = await indexFile(path);
      setDocuments([
        ...documents,
        { ...doc, status: 'indexed' as const },
      ]);
    } catch (error) {
      console.error('Failed to index file:', error);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleAddFolder = async () => {
    const path = await selectFolder();
    if (!path) return;

    setIsIndexing(true);
    try {
      const docs = await indexFolder(path);
      setDocuments([
        ...documents,
        ...docs.map((doc) => ({ ...doc, status: 'indexed' as const })),
      ]);
    } catch (error) {
      console.error('Failed to index folder:', error);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleAddWatchFolder = async () => {
    const path = await selectFolder();
    if (!path) return;

    try {
      await addWatchFolder(path);
      addWatchedFolder({
        path,
        enabled: true,
        file_count: 0,
      });
    } catch (error) {
      console.error('Failed to add watch folder:', error);
    }
  };

  const handleRemoveWatchFolder = async (path: string) => {
    try {
      await removeWatchFolder(path);
      removeWatchedFolder(path);
    } catch (error) {
      console.error('Failed to remove watch folder:', error);
    }
  };

  const handleDeleteDocument = async (id: string) => {
    try {
      await deleteDocument(id);
      removeDocFromStore(id);
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="px-4 py-3 border-b border-border">
        <div className="flex items-center justify-between mb-3">
          <h1 className="text-lg font-semibold">Documents</h1>
          <div className="flex items-center gap-2">
            <button
              onClick={loadDocuments}
              disabled={isLoading}
              className={cn(
                'p-2 rounded-lg hover:bg-muted transition-colors',
                'disabled:opacity-50'
              )}
            >
              <RefreshCw
                className={cn('w-4 h-4', isLoading && 'animate-spin')}
              />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 p-1 bg-muted rounded-lg">
          <button
            onClick={() => setActiveTab('documents')}
            className={cn(
              'flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors',
              activeTab === 'documents'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Indexed Documents
          </button>
          <button
            onClick={() => setActiveTab('folders')}
            className={cn(
              'flex-1 px-3 py-1.5 rounded text-sm font-medium transition-colors',
              activeTab === 'folders'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            )}
          >
            Watch Folders
          </button>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'documents' ? (
          <DocumentsList
            documents={documents}
            isLoading={isLoading}
            isIndexing={isIndexing}
            onAddFile={handleAddFile}
            onAddFolder={handleAddFolder}
            onDelete={handleDeleteDocument}
          />
        ) : (
          <WatchFoldersList
            folders={watchedFolders}
            onAdd={handleAddWatchFolder}
            onRemove={handleRemoveWatchFolder}
          />
        )}
      </div>
    </div>
  );
}

function DocumentsList({
  documents,
  isLoading,
  isIndexing,
  onAddFile,
  onAddFolder,
  onDelete,
}: {
  documents: Document[];
  isLoading: boolean;
  isIndexing: boolean;
  onAddFile: () => void;
  onAddFolder: () => void;
  onDelete: (id: string) => void;
}) {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={onAddFile}
          disabled={isIndexing}
          className={cn(
            'flex items-center gap-2 px-3 py-2 rounded-lg',
            'bg-primary text-primary-foreground',
            'hover:bg-primary/90 transition-colors',
            'disabled:opacity-50'
          )}
        >
          {isIndexing ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <FilePlus className="w-4 h-4" />
          )}
          Add File
        </button>
        <button
          onClick={onAddFolder}
          disabled={isIndexing}
          className={cn(
            'flex items-center gap-2 px-3 py-2 rounded-lg',
            'bg-secondary text-secondary-foreground',
            'hover:bg-secondary/80 transition-colors',
            'disabled:opacity-50'
          )}
        >
          <FolderPlus className="w-4 h-4" />
          Add Folder
        </button>
      </div>

      {/* Documents list */}
      {documents.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <FileText className="w-12 h-12 mb-4 opacity-50" />
          <p className="text-lg font-medium">No documents indexed</p>
          <p className="text-sm mt-1">Add files or folders to get started</p>
        </div>
      ) : (
        <div className="space-y-2">
          {documents.map((doc) => (
            <DocumentCard key={doc.id} document={doc} onDelete={onDelete} />
          ))}
        </div>
      )}
    </div>
  );
}

function DocumentCard({
  document,
  onDelete,
}: {
  document: Document;
  onDelete: (id: string) => void;
}) {
  const [showConfirm, setShowConfirm] = useState(false);

  return (
    <div
      className={cn(
        'flex items-center justify-between p-3 rounded-lg',
        'border border-border bg-card',
        'hover:border-primary/50 transition-colors'
      )}
    >
      <div className="flex items-center gap-3 min-w-0">
        <FileText className="w-8 h-8 text-primary flex-shrink-0" />
        <div className="min-w-0">
          <p className="font-medium truncate">{document.name}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span>{formatFileSize(document.size)}</span>
            <span>•</span>
            <span>{document.chunk_count} chunks</span>
            <span>•</span>
            <span>{formatDate(document.indexed_at)}</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {showConfirm ? (
          <>
            <button
              onClick={() => {
                onDelete(document.id);
                setShowConfirm(false);
              }}
              className="px-2 py-1 rounded text-xs bg-destructive text-destructive-foreground"
            >
              Confirm
            </button>
            <button
              onClick={() => setShowConfirm(false)}
              className="px-2 py-1 rounded text-xs bg-muted"
            >
              Cancel
            </button>
          </>
        ) : (
          <button
            onClick={() => setShowConfirm(true)}
            className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

function WatchFoldersList({
  folders,
  onAdd,
  onRemove,
}: {
  folders: WatchedFolder[];
  onAdd: () => void;
  onRemove: (path: string) => void;
}) {
  return (
    <div className="space-y-4">
      {/* Add folder button */}
      <button
        onClick={onAdd}
        className={cn(
          'flex items-center gap-2 px-3 py-2 rounded-lg',
          'bg-primary text-primary-foreground',
          'hover:bg-primary/90 transition-colors'
        )}
      >
        <FolderPlus className="w-4 h-4" />
        Add Watch Folder
      </button>

      <p className="text-sm text-muted-foreground">
        Files in watched folders are automatically indexed when they change.
      </p>

      {/* Folders list */}
      {folders.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <Folder className="w-12 h-12 mb-4 opacity-50" />
          <p className="text-lg font-medium">No watch folders</p>
          <p className="text-sm mt-1">
            Add folders to automatically index their contents
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {folders.map((folder) => (
            <div
              key={folder.path}
              className={cn(
                'flex items-center justify-between p-3 rounded-lg',
                'border border-border bg-card'
              )}
            >
              <div className="flex items-center gap-3 min-w-0">
                <Folder className="w-6 h-6 text-primary flex-shrink-0" />
                <div className="min-w-0">
                  <p className="font-medium truncate">{folder.path}</p>
                  <p className="text-xs text-muted-foreground">
                    {folder.file_count} files
                  </p>
                </div>
              </div>
              <button
                onClick={() => onRemove(folder.path)}
                className="p-2 rounded-lg hover:bg-muted transition-colors text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
