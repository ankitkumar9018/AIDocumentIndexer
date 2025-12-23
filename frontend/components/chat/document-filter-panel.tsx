"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ChevronDown,
  ChevronRight,
  FolderOpen,
  Tag,
  X,
  Filter,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface Collection {
  name: string;
  document_count: number;
}

interface DocumentFilterPanelProps {
  collections: Collection[];
  selectedCollections: string[];
  onCollectionsChange: (collections: string[]) => void;
  totalDocuments?: number;  // Actual total document count from API
  isLoading?: boolean;
  onRefresh?: () => void;
  className?: string;
}

export function DocumentFilterPanel({
  collections,
  selectedCollections,
  onCollectionsChange,
  totalDocuments: totalDocumentsProp,
  isLoading = false,
  onRefresh,
  className,
}: DocumentFilterPanelProps) {
  const [isCollectionsOpen, setIsCollectionsOpen] = useState(true);

  const handleCollectionToggle = (collectionName: string) => {
    if (selectedCollections.includes(collectionName)) {
      onCollectionsChange(selectedCollections.filter((c) => c !== collectionName));
    } else {
      onCollectionsChange([...selectedCollections, collectionName]);
    }
  };

  const handleSelectAll = () => {
    if (selectedCollections.length === collections.length) {
      onCollectionsChange([]);
    } else {
      onCollectionsChange(collections.map((c) => c.name));
    }
  };

  const handleClearFilters = () => {
    onCollectionsChange([]);
  };

  const hasActiveFilters = selectedCollections.length > 0;
  // Use the API-provided total if available, otherwise don't show a total
  const totalDocuments = totalDocumentsProp ?? 0;
  // Count filtered documents from selected collections
  const filteredDocuments = selectedCollections.length > 0
    ? collections
        .filter((c) => selectedCollections.includes(c.name))
        .reduce((sum, c) => sum + c.document_count, 0)
    : totalDocuments;

  return (
    <div className={cn("border rounded-lg bg-card", className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium text-sm">Search Scope</span>
          {hasActiveFilters && (
            <Badge variant="secondary" className="text-xs">
              {filteredDocuments} / {totalDocuments} docs
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-1">
          {hasActiveFilters && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClearFilters}
              className="h-7 px-2 text-xs"
            >
              <X className="h-3 w-3 mr-1" />
              Clear
            </Button>
          )}
          {onRefresh && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onRefresh}
              disabled={isLoading}
              className="h-7 w-7"
            >
              <RefreshCw className={cn("h-3 w-3", isLoading && "animate-spin")} />
            </Button>
          )}
        </div>
      </div>

      {/* Collections Section */}
      <Collapsible open={isCollectionsOpen} onOpenChange={setIsCollectionsOpen}>
        <div className="flex items-center justify-between w-full p-3 hover:bg-muted/50 transition-colors">
          <CollapsibleTrigger asChild>
            <button className="flex items-center gap-2 text-left flex-1">
              {isCollectionsOpen ? (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-4 w-4 text-muted-foreground" />
              )}
              <FolderOpen className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Collections</span>
              {selectedCollections.length > 0 && (
                <Badge variant="default" className="text-xs h-5">
                  {selectedCollections.length}
                </Badge>
              )}
            </button>
          </CollapsibleTrigger>
          {collections.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSelectAll}
              className="h-6 px-2 text-xs"
            >
              {selectedCollections.length === collections.length ? "None" : "All"}
            </Button>
          )}
        </div>

        <CollapsibleContent>
          <div className="max-h-64 overflow-y-auto">
            <div className="px-3 pb-3 space-y-1">
              {collections.length === 0 ? (
                <p className="text-sm text-muted-foreground py-2 text-center">
                  No collections available
                </p>
              ) : (
                collections.map((collection) => (
                  <label
                    key={collection.name}
                    className="flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-muted/50 cursor-pointer transition-colors"
                  >
                    <Checkbox
                      checked={selectedCollections.includes(collection.name)}
                      onCheckedChange={() => handleCollectionToggle(collection.name)}
                    />
                    <div className="flex items-center justify-between flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 min-w-0">
                        <Tag className="h-3 w-3 text-muted-foreground shrink-0" />
                        <span className="text-sm truncate">{collection.name}</span>
                      </div>
                      <Badge variant="outline" className="text-xs shrink-0 ml-2">
                        {collection.document_count}
                      </Badge>
                    </div>
                  </label>
                ))
              )}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>

      {/* Active Filters Summary */}
      {hasActiveFilters && (
        <div className="p-3 border-t bg-muted/30">
          <div className="flex flex-wrap gap-1">
            {selectedCollections.map((collection) => (
              <Badge
                key={collection}
                variant="secondary"
                className="text-xs cursor-pointer hover:bg-destructive hover:text-destructive-foreground transition-colors"
                onClick={() => handleCollectionToggle(collection)}
              >
                {collection}
                <X className="h-3 w-3 ml-1" />
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
