'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  useGenerationTemplates,
  useMyGenerationTemplates,
  usePopularGenerationTemplates,
  useRecordGenerationTemplateUse,
  type GenerationTemplate,
  type TemplateSettings,
} from '@/lib/api';
import {
  FileText,
  Presentation,
  ClipboardList,
  BookOpen,
  Sparkles,
  Search,
  Star,
  User,
  Clock,
  Copy,
} from 'lucide-react';

// Category icons
const categoryIcons: Record<string, React.ReactNode> = {
  report: <FileText className="h-4 w-4" />,
  proposal: <ClipboardList className="h-4 w-4" />,
  presentation: <Presentation className="h-4 w-4" />,
  meeting_notes: <ClipboardList className="h-4 w-4" />,
  documentation: <BookOpen className="h-4 w-4" />,
  custom: <Sparkles className="h-4 w-4" />,
};

// Format icons
const formatIcons: Record<string, string> = {
  pptx: 'PPTX',
  docx: 'DOCX',
  pdf: 'PDF',
  md: 'MD',
  html: 'HTML',
  xlsx: 'XLSX',
};

interface TemplateSelectorProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (template: GenerationTemplate) => void;
}

export function TemplateSelector({
  open,
  onOpenChange,
  onSelect,
}: TemplateSelectorProps) {
  const [activeTab, setActiveTab] = useState<'all' | 'popular' | 'mine'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | undefined>();

  // Fetch templates
  const { data: allTemplates, isLoading: isLoadingAll } = useGenerationTemplates({
    category: selectedCategory,
    search: searchQuery || undefined,
    include_system: true,
    include_public: true,
    limit: 50,
  });

  const { data: popularTemplates, isLoading: isLoadingPopular } = usePopularGenerationTemplates(
    selectedCategory,
    10
  );

  const { data: myTemplates, isLoading: isLoadingMine } = useMyGenerationTemplates(50);

  const recordUse = useRecordGenerationTemplateUse();

  const handleSelect = (template: GenerationTemplate) => {
    // Record template usage
    recordUse.mutate(template.id);
    onSelect(template);
    onOpenChange(false);
  };

  const categories = [
    { value: undefined, label: 'All' },
    { value: 'report', label: 'Reports' },
    { value: 'proposal', label: 'Proposals' },
    { value: 'presentation', label: 'Presentations' },
    { value: 'meeting_notes', label: 'Meeting Notes' },
    { value: 'documentation', label: 'Documentation' },
    { value: 'custom', label: 'Custom' },
  ];

  const renderTemplateCard = (template: GenerationTemplate) => {
    const settings = template.settings as TemplateSettings;

    return (
      <Card
        key={template.id}
        className="cursor-pointer transition-all hover:shadow-md hover:border-primary/50"
        onClick={() => handleSelect(template)}
      >
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-2">
              {categoryIcons[template.category] || <Sparkles className="h-4 w-4" />}
              <CardTitle className="text-base">{template.name}</CardTitle>
            </div>
            <div className="flex items-center gap-1">
              {template.is_system && (
                <Badge variant="secondary" className="text-xs">
                  <Star className="h-3 w-3 mr-1" />
                  System
                </Badge>
              )}
              {!template.is_system && template.is_public && (
                <Badge variant="outline" className="text-xs">
                  Public
                </Badge>
              )}
            </div>
          </div>
          <CardDescription className="text-xs line-clamp-2">
            {template.description || 'No description'}
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="outline" className="text-xs">
              {formatIcons[settings.output_format] || settings.output_format?.toUpperCase()}
            </Badge>
            <Badge variant="outline" className="text-xs capitalize">
              {settings.theme}
            </Badge>
            {settings.include_toc && (
              <Badge variant="outline" className="text-xs">
                TOC
              </Badge>
            )}
            {settings.enable_animations && (
              <Badge variant="outline" className="text-xs">
                Animations
              </Badge>
            )}
          </div>
          {template.use_count > 0 && (
            <div className="flex items-center gap-1 mt-2 text-xs text-muted-foreground">
              <Clock className="h-3 w-3" />
              Used {template.use_count} times
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderTemplateList = (
    templates: GenerationTemplate[] | undefined,
    isLoading: boolean,
    emptyMessage: string
  ) => {
    if (isLoading) {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="pb-2">
                <div className="h-5 bg-muted rounded w-3/4" />
                <div className="h-3 bg-muted rounded w-full mt-2" />
              </CardHeader>
              <CardContent className="pt-0">
                <div className="flex gap-1">
                  <div className="h-5 bg-muted rounded w-12" />
                  <div className="h-5 bg-muted rounded w-16" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      );
    }

    if (!templates || templates.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
          <Copy className="h-8 w-8 mb-2 opacity-50" />
          <p>{emptyMessage}</p>
        </div>
      );
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {templates.map(renderTemplateCard)}
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Choose a Template</DialogTitle>
          <DialogDescription>
            Start with a pre-configured template to speed up document creation
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Search and Filter */}
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>

          {/* Category Filter */}
          <div className="flex flex-wrap gap-1.5">
            {categories.map((cat) => (
              <Button
                key={cat.value ?? 'all'}
                variant={selectedCategory === cat.value ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedCategory(cat.value)}
                className="text-xs"
              >
                {cat.label}
              </Button>
            ))}
          </div>

          {/* Tabs */}
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="all" className="flex items-center gap-1">
                <Sparkles className="h-3.5 w-3.5" />
                All Templates
              </TabsTrigger>
              <TabsTrigger value="popular" className="flex items-center gap-1">
                <Star className="h-3.5 w-3.5" />
                Popular
              </TabsTrigger>
              <TabsTrigger value="mine" className="flex items-center gap-1">
                <User className="h-3.5 w-3.5" />
                My Templates
              </TabsTrigger>
            </TabsList>

            <ScrollArea className="h-[400px] mt-4 pr-4">
              <TabsContent value="all" className="mt-0">
                {renderTemplateList(
                  allTemplates?.templates,
                  isLoadingAll,
                  'No templates found. Try adjusting your search.'
                )}
              </TabsContent>

              <TabsContent value="popular" className="mt-0">
                {renderTemplateList(
                  popularTemplates,
                  isLoadingPopular,
                  'No popular templates yet.'
                )}
              </TabsContent>

              <TabsContent value="mine" className="mt-0">
                {renderTemplateList(
                  myTemplates,
                  isLoadingMine,
                  'You haven\'t created any templates yet. Save your current settings as a template!'
                )}
              </TabsContent>
            </ScrollArea>
          </Tabs>
        </div>

        <div className="flex justify-between items-center pt-4 border-t">
          <p className="text-sm text-muted-foreground">
            Or start from scratch with default settings
          </p>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
