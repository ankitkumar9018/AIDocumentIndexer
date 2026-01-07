'use client';

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import {
  useCreateGenerationTemplate,
  type TemplateSettings,
  type TemplateCategory,
} from '@/lib/api';
import { Save, X, Loader2 } from 'lucide-react';

interface SaveTemplateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentSettings: TemplateSettings;
  defaultCollections?: string[];
  onSaved?: (templateId: string) => void;
}

const categories: { value: TemplateCategory; label: string }[] = [
  { value: 'report', label: 'Report' },
  { value: 'proposal', label: 'Proposal' },
  { value: 'presentation', label: 'Presentation' },
  { value: 'meeting_notes', label: 'Meeting Notes' },
  { value: 'documentation', label: 'Documentation' },
  { value: 'custom', label: 'Custom' },
];

export function SaveTemplateDialog({
  open,
  onOpenChange,
  currentSettings,
  defaultCollections,
  onSaved,
}: SaveTemplateDialogProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState<TemplateCategory>('custom');
  const [isPublic, setIsPublic] = useState(false);
  const [tags, setTags] = useState<string[]>([]);
  const [tagInput, setTagInput] = useState('');

  const createTemplate = useCreateGenerationTemplate();

  const handleAddTag = () => {
    const trimmedTag = tagInput.trim().toLowerCase();
    if (trimmedTag && !tags.includes(trimmedTag)) {
      setTags([...tags, trimmedTag]);
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(tags.filter((t) => t !== tagToRemove));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAddTag();
    }
  };

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error('Please enter a name for your template.');
      return;
    }

    try {
      const template = await createTemplate.mutateAsync({
        name: name.trim(),
        description: description.trim() || undefined,
        category,
        settings: currentSettings,
        default_collections: defaultCollections,
        is_public: isPublic,
        tags: tags.length > 0 ? tags : undefined,
      });

      toast.success(`"${template.name}" has been saved to your templates.`);

      onSaved?.(template.id);
      onOpenChange(false);

      // Reset form
      setName('');
      setDescription('');
      setCategory('custom');
      setIsPublic(false);
      setTags([]);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to save template');
    }
  };

  // Generate preview of settings
  const settingsPreview = [
    { label: 'Format', value: currentSettings.output_format?.toUpperCase() },
    { label: 'Theme', value: currentSettings.theme },
    currentSettings.font_family && { label: 'Font', value: currentSettings.font_family },
    currentSettings.layout_template && { label: 'Layout', value: currentSettings.layout_template },
    currentSettings.include_toc && { label: 'TOC', value: 'Yes' },
    currentSettings.include_sources && { label: 'Sources', value: 'Yes' },
    currentSettings.use_existing_docs && { label: 'Style Learning', value: 'Yes' },
    currentSettings.enable_animations && { label: 'Animations', value: 'Yes' },
  ].filter(Boolean);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Save as Template</DialogTitle>
          <DialogDescription>
            Save your current settings as a reusable template
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Name */}
          <div className="space-y-2">
            <Label htmlFor="template-name">Template Name *</Label>
            <Input
              id="template-name"
              placeholder="e.g., Quarterly Report Template"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>

          {/* Description */}
          <div className="space-y-2">
            <Label htmlFor="template-description">Description</Label>
            <Textarea
              id="template-description"
              placeholder="Describe when to use this template..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={2}
            />
          </div>

          {/* Category */}
          <div className="space-y-2">
            <Label htmlFor="template-category">Category</Label>
            <Select value={category} onValueChange={(v) => setCategory(v as TemplateCategory)}>
              <SelectTrigger id="template-category">
                <SelectValue placeholder="Select category" />
              </SelectTrigger>
              <SelectContent>
                {categories.map((cat) => (
                  <SelectItem key={cat.value} value={cat.value}>
                    {cat.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Tags */}
          <div className="space-y-2">
            <Label htmlFor="template-tags">Tags</Label>
            <div className="flex gap-2">
              <Input
                id="template-tags"
                placeholder="Add a tag..."
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleKeyDown}
              />
              <Button type="button" variant="outline" size="sm" onClick={handleAddTag}>
                Add
              </Button>
            </div>
            {tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                    <button
                      type="button"
                      onClick={() => handleRemoveTag(tag)}
                      className="ml-1 hover:text-destructive"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Public toggle */}
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="template-public">Make Public</Label>
              <p className="text-xs text-muted-foreground">
                Allow other users to use this template
              </p>
            </div>
            <Switch
              id="template-public"
              checked={isPublic}
              onCheckedChange={setIsPublic}
            />
          </div>

          {/* Settings Preview */}
          <div className="space-y-2">
            <Label>Included Settings</Label>
            <div className="flex flex-wrap gap-1.5 p-2 bg-muted/50 rounded-md">
              {settingsPreview.map((item, i) => (
                item && (
                  <Badge key={i} variant="outline" className="text-xs">
                    {item.label}: {item.value}
                  </Badge>
                )
              ))}
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={createTemplate.isPending}>
            {createTemplate.isPending ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="mr-2 h-4 w-4" />
                Save Template
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
