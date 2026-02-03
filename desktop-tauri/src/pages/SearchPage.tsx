import { useState } from 'react';
import { Search, Loader2, FileText, ExternalLink } from 'lucide-react';
import { useAppStore } from '../lib/store';
import { searchLocal, searchServer, type SearchResult } from '../lib/tauri';
import { cn, formatFileSize } from '../lib/utils';

export function SearchPage() {
  const { mode, searchQuery, setSearchQuery, searchResults, setSearchResults } =
    useAppStore();
  const [isSearching, setIsSearching] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim() || isSearching) return;

    setIsSearching(true);
    try {
      const results =
        mode === 'local'
          ? await searchLocal(searchQuery.trim())
          : await searchServer(searchQuery.trim());
      setSearchResults(results);
    } catch (error) {
      console.error('Search failed:', error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="px-4 py-3 border-b border-border">
        <h1 className="text-lg font-semibold mb-3">Search Documents</h1>

        {/* Search form */}
        <form onSubmit={handleSearch} className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search your documents..."
              className={cn(
                'w-full pl-10 pr-4 py-2 rounded-lg',
                'bg-muted border border-input',
                'focus:outline-none focus:ring-2 focus:ring-ring',
                'placeholder:text-muted-foreground'
              )}
            />
          </div>
          <button
            type="submit"
            disabled={!searchQuery.trim() || isSearching}
            className={cn(
              'px-4 py-2 rounded-lg',
              'bg-primary text-primary-foreground',
              'hover:bg-primary/90 transition-colors',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {isSearching ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              'Search'
            )}
          </button>
        </form>
      </header>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-4">
        {searchResults.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <Search className="w-12 h-12 mb-4 opacity-50" />
            <p className="text-lg font-medium">Search your knowledge base</p>
            <p className="text-sm mt-1">
              Find relevant information across all your documents
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Found {searchResults.length} results for "{searchQuery}"
            </p>
            {searchResults.map((result, index) => (
              <SearchResultCard key={index} result={result} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SearchResultCard({ result }: { result: SearchResult }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={cn(
        'p-4 rounded-lg border border-border bg-card',
        'hover:border-primary/50 transition-colors'
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary" />
          <span className="font-medium truncate">{result.document_name}</span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'px-2 py-0.5 rounded text-xs font-medium',
              result.score >= 0.8
                ? 'bg-green-500/20 text-green-600 dark:text-green-400'
                : result.score >= 0.6
                  ? 'bg-yellow-500/20 text-yellow-600 dark:text-yellow-400'
                  : 'bg-muted text-muted-foreground'
            )}
          >
            {(result.score * 100).toFixed(0)}% match
          </span>
        </div>
      </div>

      {/* Content preview */}
      <p
        className={cn(
          'text-sm text-muted-foreground',
          !expanded && 'line-clamp-3'
        )}
      >
        {result.content}
      </p>

      {/* Actions */}
      <div className="flex items-center gap-4 mt-3">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-primary hover:underline"
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      </div>
    </div>
  );
}
