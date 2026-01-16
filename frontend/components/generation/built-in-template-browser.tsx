'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Presentation,
  FileText,
  FileSpreadsheet,
  ExternalLink,
  Download,
  Search,
  Loader2,
  Check,
  Palette,
  FolderOpen,
} from 'lucide-react';
import { api } from '@/lib/api';
import type {
  DocumentTemplateMetadata,
  DocumentTemplatesByType,
  ExternalTemplateSource
} from '@/lib/api';
import { toast } from 'sonner';

interface BuiltInTemplateBrowserProps {
  fileType?: 'pptx' | 'docx' | 'xlsx';
  onSelectTemplate?: (template: DocumentTemplateMetadata) => void;
  selectedTemplateId?: string | null;
  showExternalSources?: boolean;
}

const fileTypeIcons = {
  pptx: Presentation,
  docx: FileText,
  xlsx: FileSpreadsheet,
};

const fileTypeLabels = {
  pptx: 'Presentations',
  docx: 'Documents',
  xlsx: 'Spreadsheets',
};

export function BuiltInTemplateBrowser({
  fileType,
  onSelectTemplate,
  selectedTemplateId,
  showExternalSources = true,
}: BuiltInTemplateBrowserProps) {
  const [templates, setTemplates] = useState<DocumentTemplatesByType[]>([]);
  const [externalSources, setExternalSources] = useState<ExternalTemplateSource[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<string>(fileType || 'pptx');
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const [templatesData, sourcesData] = await Promise.all([
          api.listDocumentTemplatesByType(),
          showExternalSources ? api.getExternalTemplateSources() : Promise.resolve({ sources: [] }),
        ]);
        setTemplates(templatesData);
        setExternalSources(sourcesData.sources);
      } catch (error) {
        console.error('Failed to load templates:', error);
        toast.error('Failed to load templates');
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [showExternalSources]);

  // Filter templates based on search and file type
  const filteredTemplates = templates
    .filter(t => !fileType || t.file_type === fileType)
    .map(typeGroup => ({
      ...typeGroup,
      categories: typeGroup.categories.map(cat => ({
        ...cat,
        templates: cat.templates.filter(template =>
          !searchQuery ||
          template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          template.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
          template.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
        ),
      })).filter(cat => cat.templates.length > 0),
    }))
    .filter(t => t.categories.length > 0);

  const filteredSources = externalSources.filter(
    source => !fileType || source.file_types.includes(fileType)
  );

  const handleSelectTemplate = (template: DocumentTemplateMetadata) => {
    onSelectTemplate?.(template);
    setIsDialogOpen(false);
    toast.success(`Selected template: ${template.name}`);
  };

  const handleDownloadTemplate = (template: DocumentTemplateMetadata) => {
    const downloadUrl = api.getDocumentTemplateDownloadUrl(
      template.file_type,
      template.category,
      template.id
    );
    window.open(downloadUrl, '_blank');
    toast.success('Template download started');
  };

  const renderTemplateCard = (template: DocumentTemplateMetadata) => {
    const isSelected = selectedTemplateId === template.id;
    const Icon = fileTypeIcons[template.file_type as keyof typeof fileTypeIcons] || FileText;

    return (
      <Card
        key={template.id}
        className={`cursor-pointer transition-all hover:shadow-md ${
          isSelected
            ? 'border-primary ring-2 ring-primary/20'
            : 'hover:border-primary/50'
        }`}
        onClick={() => handleSelectTemplate(template)}
      >
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
              style={{ backgroundColor: template.primary_color + '20' }}
            >
              <Icon className="h-5 w-5" style={{ color: template.primary_color }} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h4 className="font-medium text-sm truncate">{template.name}</h4>
                {isSelected && (
                  <Check className="h-4 w-4 text-primary shrink-0" />
                )}
              </div>
              <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
                {template.description}
              </p>
              <div className="flex items-center gap-2 mt-2 flex-wrap">
                <Badge variant="secondary" className="text-[10px]">
                  {template.style}
                </Badge>
                {template.recommended_slides && (
                  <span className="text-[10px] text-muted-foreground">
                    ~{template.recommended_slides} slides
                  </span>
                )}
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0"
              onClick={(e) => {
                e.stopPropagation();
                handleDownloadTemplate(template);
              }}
            >
              <Download className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderExternalSourceCard = (source: ExternalTemplateSource) => (
    <Card key={source.name} className="hover:shadow-md transition-all">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h4 className="font-medium text-sm">{source.name}</h4>
            <p className="text-xs text-muted-foreground mt-1">{source.description}</p>
            <div className="flex gap-1 mt-2">
              {source.file_types.map(ft => (
                <Badge key={ft} variant="outline" className="text-[10px]">
                  {ft.toUpperCase()}
                </Badge>
              ))}
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            asChild
          >
            <a href={source.url} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="h-3 w-3 mr-1" />
              Visit
            </a>
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const content = (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search templates..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-9"
        />
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      ) : (
        <>
          {/* Template Type Tabs (only if not filtered to single type) */}
          {!fileType ? (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="w-full">
                <TabsTrigger value="pptx" className="flex-1">
                  <Presentation className="h-4 w-4 mr-2" />
                  Presentations
                </TabsTrigger>
                <TabsTrigger value="docx" className="flex-1">
                  <FileText className="h-4 w-4 mr-2" />
                  Documents
                </TabsTrigger>
                <TabsTrigger value="xlsx" className="flex-1">
                  <FileSpreadsheet className="h-4 w-4 mr-2" />
                  Spreadsheets
                </TabsTrigger>
              </TabsList>

              {['pptx', 'docx', 'xlsx'].map(type => (
                <TabsContent key={type} value={type} className="mt-4">
                  <ScrollArea className="h-[400px]">
                    {renderTemplatesForType(type)}
                  </ScrollArea>
                </TabsContent>
              ))}
            </Tabs>
          ) : (
            <ScrollArea className="h-[400px]">
              {renderTemplatesForType(fileType)}
            </ScrollArea>
          )}

          {/* External Sources Section */}
          {showExternalSources && filteredSources.length > 0 && (
            <div className="pt-4 border-t">
              <div className="flex items-center gap-2 mb-3">
                <ExternalLink className="h-4 w-4 text-muted-foreground" />
                <h3 className="text-sm font-medium">More Templates Online</h3>
              </div>
              <div className="grid gap-3">
                {filteredSources.slice(0, 4).map(renderExternalSourceCard)}
              </div>
              {filteredSources.length > 4 && (
                <Button variant="link" className="w-full mt-2 text-xs">
                  View all {filteredSources.length} sources
                </Button>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );

  function renderTemplatesForType(type: string) {
    const typeData = filteredTemplates.find(t => t.file_type === type);

    if (!typeData || typeData.categories.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <FolderOpen className="h-12 w-12 text-muted-foreground mb-3" />
          <p className="text-sm text-muted-foreground">
            {searchQuery ? 'No templates match your search' : 'No templates available'}
          </p>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        {typeData.categories.map(category => (
          <div key={category.name}>
            <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
              <Palette className="h-4 w-4" />
              {category.display_name}
            </h3>
            <div className="grid gap-3 sm:grid-cols-2">
              {category.templates.map(renderTemplateCard)}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // If used inline (not in dialog), return content directly
  if (!onSelectTemplate) {
    return content;
  }

  // Dialog mode for template selection
  return (
    <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <FolderOpen className="h-4 w-4 mr-2" />
          Browse Built-in Templates
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Palette className="h-5 w-5" />
            Built-in Templates
          </DialogTitle>
          <DialogDescription>
            Choose from our collection of professionally designed templates
          </DialogDescription>
        </DialogHeader>
        <div className="flex-1 overflow-hidden">
          {content}
        </div>
      </DialogContent>
    </Dialog>
  );
}

// Compact inline version for embedding in forms
export function BuiltInTemplateSelector({
  fileType,
  onSelectTemplate,
  selectedTemplateId,
}: {
  fileType: 'pptx' | 'docx' | 'xlsx';
  onSelectTemplate: (template: DocumentTemplateMetadata) => void;
  selectedTemplateId?: string | null;
}) {
  const [templates, setTemplates] = useState<DocumentTemplateMetadata[]>([]);
  const [externalSources, setExternalSources] = useState<ExternalTemplateSource[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Ensure we only render on client to avoid hydration mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const fetchData = async () => {
      setIsLoading(true);
      try {
        const [templatesData, sourcesData] = await Promise.all([
          api.listDocumentTemplates({ fileType }),
          api.getExternalTemplateSources(fileType),
        ]);
        setTemplates(templatesData.templates);
        setExternalSources(sourcesData.sources);
      } catch (error) {
        console.error('Failed to load templates:', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [fileType, mounted]);

  // Don't render anything on server to avoid hydration mismatch
  if (!mounted) {
    return null;
  }

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 p-3 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading built-in templates...
      </div>
    );
  }

  if (templates.length === 0) {
    return null;
  }

  const displayedTemplates = showAll ? templates : templates.slice(0, 4);
  const Icon = fileTypeIcons[fileType];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">Built-in Templates</span>
          <Badge variant="secondary" className="text-[10px]">
            {templates.length}
          </Badge>
        </div>
        {templates.length > 4 && (
          <Button
            variant="link"
            size="sm"
            className="text-xs h-auto p-0"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? 'Show less' : `View all ${templates.length}`}
          </Button>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2">
        {displayedTemplates.map((template) => {
          const isSelected = selectedTemplateId === template.id;
          return (
            <div
              key={template.id}
              onClick={() => onSelectTemplate(template)}
              className={`p-3 rounded-lg border cursor-pointer transition-all ${
                isSelected
                  ? 'border-primary ring-2 ring-primary/20'
                  : 'border-border hover:border-primary/50'
              }`}
            >
              <div className="flex items-center gap-2">
                <div
                  className="w-6 h-6 rounded flex items-center justify-center shrink-0"
                  style={{ backgroundColor: template.primary_color + '20' }}
                >
                  <div
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: template.primary_color }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{template.name}</p>
                  <p className="text-xs text-muted-foreground truncate">
                    {template.category} • {template.style}
                  </p>
                </div>
                {isSelected && (
                  <Check className="h-4 w-4 text-primary shrink-0" />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* External sources link */}
      {externalSources.length > 0 && (
        <div className="flex items-center justify-end">
          <a
            href={externalSources[0].url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-muted-foreground hover:text-primary flex items-center gap-1"
          >
            <ExternalLink className="h-3 w-3" />
            Get more templates from {externalSources[0].name}
          </a>
        </div>
      )}
    </div>
  );
}
