"use client";

import { useState, useMemo, useCallback } from "react";
import {
  Plus,
  Search,
  Star,
  MoreVertical,
  Copy,
  Trash2,
  Edit,
  Eye,
  Sparkles,
  FileText,
  Code,
  BarChart3,
  MessageSquare,
  Wand2,
  Loader2,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import {
  usePromptTemplates,
  usePromptTemplate,
  useTemplateCategories,
  useCreatePromptTemplate,
  useUpdatePromptTemplate,
  useDeletePromptTemplate,
  useApplyPromptTemplate,
  useDuplicatePromptTemplate,
} from "@/lib/api";
import type { TemplateListItem, TemplateVariable } from "@/lib/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  general: <FileText className="h-5 w-5" />,
  coding: <Code className="h-5 w-5" />,
  analysis: <BarChart3 className="h-5 w-5" />,
  writing: <Edit className="h-5 w-5" />,
  chat: <MessageSquare className="h-5 w-5" />,
  creative: <Sparkles className="h-5 w-5" />,
};

function getCategoryIcon(category: string) {
  return CATEGORY_ICONS[category.toLowerCase()] ?? <Wand2 className="h-5 w-5" />;
}

const EMPTY_FORM: TemplateFormData = {
  name: "",
  description: "",
  category: "general",
  prompt_text: "",
  system_prompt: "",
  is_public: false,
  tags: "",
  variables: [],
};

interface TemplateFormData {
  name: string;
  description: string;
  category: string;
  prompt_text: string;
  system_prompt: string;
  is_public: boolean;
  tags: string;
  variables: TemplateVariable[];
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TemplateFormDialog({
  open,
  onOpenChange,
  title,
  description,
  initial,
  onSubmit,
  isSubmitting,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  initial: TemplateFormData;
  onSubmit: (data: TemplateFormData) => void;
  isSubmitting: boolean;
}) {
  const [form, setForm] = useState<TemplateFormData>(initial);

  // Reset form when dialog opens with new initial values
  const handleOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (nextOpen) setForm(initial);
      onOpenChange(nextOpen);
    },
    [initial, onOpenChange]
  );

  const handleAddVariable = () => {
    setForm((f) => ({
      ...f,
      variables: [
        ...f.variables,
        { name: "", description: "", default: "", required: false },
      ],
    }));
  };

  const handleRemoveVariable = (idx: number) => {
    setForm((f) => ({
      ...f,
      variables: f.variables.filter((_, i) => i !== idx),
    }));
  };

  const handleVariableChange = (
    idx: number,
    field: keyof TemplateVariable,
    value: string | boolean
  ) => {
    setForm((f) => ({
      ...f,
      variables: f.variables.map((v, i) =>
        i === idx ? { ...v, [field]: value } : v
      ),
    }));
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[640px] max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        <ScrollArea className="flex-1 pr-4 -mr-4">
          <div className="space-y-4 py-2">
            {/* Name */}
            <div className="space-y-2">
              <Label htmlFor="tpl-name">Name *</Label>
              <Input
                id="tpl-name"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
                placeholder="My Prompt Template"
              />
            </div>

            {/* Description */}
            <div className="space-y-2">
              <Label htmlFor="tpl-desc">Description</Label>
              <Textarea
                id="tpl-desc"
                value={form.description}
                onChange={(e) =>
                  setForm({ ...form, description: e.target.value })
                }
                placeholder="Describe what this template does..."
                rows={2}
              />
            </div>

            {/* Category */}
            <div className="space-y-2">
              <Label htmlFor="tpl-cat">Category</Label>
              <Input
                id="tpl-cat"
                value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}
                placeholder="general"
              />
            </div>

            {/* Tags */}
            <div className="space-y-2">
              <Label htmlFor="tpl-tags">Tags (comma-separated)</Label>
              <Input
                id="tpl-tags"
                value={form.tags}
                onChange={(e) => setForm({ ...form, tags: e.target.value })}
                placeholder="summarize, document, ai"
              />
            </div>

            {/* Prompt text */}
            <div className="space-y-2">
              <Label htmlFor="tpl-prompt">Prompt Text *</Label>
              <Textarea
                id="tpl-prompt"
                value={form.prompt_text}
                onChange={(e) =>
                  setForm({ ...form, prompt_text: e.target.value })
                }
                placeholder="Write your prompt here. Use {{variable_name}} for variables."
                rows={5}
                className="font-mono text-sm"
              />
            </div>

            {/* System prompt */}
            <div className="space-y-2">
              <Label htmlFor="tpl-sys">System Prompt</Label>
              <Textarea
                id="tpl-sys"
                value={form.system_prompt}
                onChange={(e) =>
                  setForm({ ...form, system_prompt: e.target.value })
                }
                placeholder="Optional system prompt..."
                rows={3}
                className="font-mono text-sm"
              />
            </div>

            {/* Public toggle */}
            <div className="flex items-center justify-between">
              <div>
                <Label>Public</Label>
                <p className="text-xs text-muted-foreground">
                  Make this template visible to all users
                </p>
              </div>
              <Switch
                checked={form.is_public}
                onCheckedChange={(checked) =>
                  setForm({ ...form, is_public: checked })
                }
              />
            </div>

            {/* Variables */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Variables</Label>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleAddVariable}
                >
                  <Plus className="h-3 w-3 mr-1" />
                  Add Variable
                </Button>
              </div>
              {form.variables.map((v, idx) => (
                <div
                  key={idx}
                  className="rounded-lg border p-3 space-y-2 relative"
                >
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2 h-6 w-6"
                    onClick={() => handleRemoveVariable(idx)}
                  >
                    <X className="h-3 w-3" />
                  </Button>
                  <div className="grid grid-cols-2 gap-2">
                    <div>
                      <Label className="text-xs">Name</Label>
                      <Input
                        value={v.name}
                        onChange={(e) =>
                          handleVariableChange(idx, "name", e.target.value)
                        }
                        placeholder="variable_name"
                        className="h-8 text-sm"
                      />
                    </div>
                    <div>
                      <Label className="text-xs">Default</Label>
                      <Input
                        value={v.default}
                        onChange={(e) =>
                          handleVariableChange(idx, "default", e.target.value)
                        }
                        placeholder="Default value"
                        className="h-8 text-sm"
                      />
                    </div>
                  </div>
                  <div>
                    <Label className="text-xs">Description</Label>
                    <Input
                      value={v.description}
                      onChange={(e) =>
                        handleVariableChange(idx, "description", e.target.value)
                      }
                      placeholder="What this variable represents"
                      className="h-8 text-sm"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch
                      checked={v.required}
                      onCheckedChange={(checked) =>
                        handleVariableChange(idx, "required", checked)
                      }
                    />
                    <Label className="text-xs">Required</Label>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </ScrollArea>
        <DialogFooter className="pt-4">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button
            onClick={() => onSubmit(form)}
            disabled={isSubmitting || !form.name.trim() || !form.prompt_text.trim()}
          >
            {isSubmitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            {title.startsWith("Edit") ? "Save Changes" : "Create Template"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function UseTemplateDialog({
  open,
  onOpenChange,
  templateId,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  templateId: string;
}) {
  const { data: template, isLoading } = usePromptTemplate(templateId, {
    enabled: open && !!templateId,
  });
  const applyMutation = useApplyPromptTemplate();
  const [variableValues, setVariableValues] = useState<Record<string, string>>(
    {}
  );
  const [appliedText, setAppliedText] = useState<string | null>(null);

  const handleApply = async () => {
    try {
      const result = await applyMutation.mutateAsync({
        templateId,
        variables: variableValues,
      });
      setAppliedText(result.prompt_text);
    } catch {
      toast.error("Failed to apply template");
    }
  };

  const handleCopy = async () => {
    const textToCopy = appliedText ?? template?.prompt_text ?? "";
    try {
      await navigator.clipboard.writeText(textToCopy);
      toast.success("Prompt copied to clipboard");
    } catch {
      toast.error("Failed to copy to clipboard");
    }
  };

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      setVariableValues({});
      setAppliedText(null);
    }
    onOpenChange(nextOpen);
  };

  const hasVariables = template?.variables && template.variables.length > 0;

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Wand2 className="h-5 w-5" />
            {template?.name ?? "Use Template"}
          </DialogTitle>
          <DialogDescription>
            {template?.description ?? "Fill in any variables and copy the prompt."}
          </DialogDescription>
        </DialogHeader>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : template ? (
          <ScrollArea className="flex-1 pr-4 -mr-4">
            <div className="space-y-4 py-2">
              {/* Variable inputs */}
              {hasVariables && (
                <div className="space-y-3">
                  <Label className="text-sm font-medium">Variables</Label>
                  {template.variables.map((v) => (
                    <div key={v.name} className="space-y-1">
                      <Label className="text-xs">
                        {v.name}
                        {v.required && (
                          <span className="text-destructive ml-1">*</span>
                        )}
                      </Label>
                      {v.description && (
                        <p className="text-xs text-muted-foreground">
                          {v.description}
                        </p>
                      )}
                      <Input
                        value={variableValues[v.name] ?? v.default ?? ""}
                        onChange={(e) =>
                          setVariableValues((prev) => ({
                            ...prev,
                            [v.name]: e.target.value,
                          }))
                        }
                        placeholder={v.default || `Enter ${v.name}`}
                        className="h-8 text-sm"
                      />
                    </div>
                  ))}
                </div>
              )}

              {/* System prompt */}
              {template.system_prompt && (
                <div className="space-y-1">
                  <Label className="text-xs text-muted-foreground">
                    System Prompt
                  </Label>
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <pre className="text-xs whitespace-pre-wrap font-mono">
                      {template.system_prompt}
                    </pre>
                  </div>
                </div>
              )}

              {/* Prompt text / Applied text */}
              <div className="space-y-1">
                <Label className="text-xs text-muted-foreground">
                  {appliedText ? "Applied Prompt" : "Prompt Text"}
                </Label>
                <div className="rounded-lg border bg-muted/30 p-3">
                  <pre className="text-sm whitespace-pre-wrap font-mono">
                    {appliedText ?? template.prompt_text}
                  </pre>
                </div>
              </div>
            </div>
          </ScrollArea>
        ) : null}

        <DialogFooter className="pt-4">
          {hasVariables && !appliedText && (
            <Button
              variant="outline"
              onClick={handleApply}
              disabled={applyMutation.isPending}
            >
              {applyMutation.isPending && (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              )}
              Apply Variables
            </Button>
          )}
          <Button onClick={handleCopy}>
            <Copy className="h-4 w-4 mr-2" />
            Copy to Clipboard
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function DeleteConfirmDialog({
  open,
  onOpenChange,
  templateName,
  onConfirm,
  isDeleting,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  templateName: string;
  onConfirm: () => void;
  isDeleting: boolean;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[420px]">
        <DialogHeader>
          <DialogTitle>Delete Template</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete &quot;{templateName}&quot;? This
            action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// Main Page
// ---------------------------------------------------------------------------

export default function PromptLibraryPage() {
  // Filters
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  // Dialog state
  const [createOpen, setCreateOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [editTemplateId, setEditTemplateId] = useState<string | null>(null);
  const [useOpen, setUseOpen] = useState(false);
  const [useTemplateId, setUseTemplateId] = useState<string>("");
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);

  // Queries
  const {
    data: templates,
    isLoading: templatesLoading,
  } = usePromptTemplates({
    category: selectedCategory !== "all" ? selectedCategory : undefined,
    search: searchQuery || undefined,
  });
  const { data: categories } = useTemplateCategories();

  // The edit form needs the full template detail
  const { data: editTemplate } = usePromptTemplate(editTemplateId ?? "", {
    enabled: !!editTemplateId,
  });

  // Mutations
  const createMutation = useCreatePromptTemplate();
  const updateMutation = useUpdatePromptTemplate();
  const deleteMutation = useDeletePromptTemplate();
  const duplicateMutation = useDuplicatePromptTemplate();

  // Derive edit form initial data from the loaded template
  const editFormInitial = useMemo<TemplateFormData>(() => {
    if (!editTemplate) return EMPTY_FORM;
    return {
      name: editTemplate.name,
      description: editTemplate.description ?? "",
      category: editTemplate.category,
      prompt_text: editTemplate.prompt_text,
      system_prompt: editTemplate.system_prompt ?? "",
      is_public: editTemplate.is_public,
      tags: (editTemplate.tags ?? []).join(", "),
      variables: editTemplate.variables ?? [],
    };
  }, [editTemplate]);

  // Handlers ----------------------------------------------------------------

  const handleCreate = async (data: TemplateFormData) => {
    try {
      const tags = data.tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      await createMutation.mutateAsync({
        name: data.name,
        description: data.description || undefined,
        category: data.category || "general",
        prompt_text: data.prompt_text,
        system_prompt: data.system_prompt || undefined,
        is_public: data.is_public,
        tags: tags.length > 0 ? tags : undefined,
      });
      toast.success("Template created");
      setCreateOpen(false);
    } catch {
      toast.error("Failed to create template");
    }
  };

  const handleUpdate = async (data: TemplateFormData) => {
    if (!editTemplateId) return;
    try {
      const tags = data.tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      await updateMutation.mutateAsync({
        templateId: editTemplateId,
        data: {
          name: data.name,
          description: data.description || undefined,
          category: data.category || "general",
          prompt_text: data.prompt_text,
          system_prompt: data.system_prompt || undefined,
          is_public: data.is_public,
          tags: tags.length > 0 ? tags : undefined,
          variables: data.variables.length > 0 ? data.variables : undefined,
        },
      });
      toast.success("Template updated");
      setEditOpen(false);
      setEditTemplateId(null);
    } catch {
      toast.error("Failed to update template");
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteMutation.mutateAsync(deleteTarget.id);
      toast.success("Template deleted");
      setDeleteOpen(false);
      setDeleteTarget(null);
    } catch {
      toast.error("Failed to delete template");
    }
  };

  const handleDuplicate = async (template: TemplateListItem) => {
    try {
      await duplicateMutation.mutateAsync({
        templateId: template.id,
        newName: `${template.name} (Copy)`,
      });
      toast.success("Template duplicated");
    } catch {
      toast.error("Failed to duplicate template");
    }
  };

  const openEdit = (id: string) => {
    setEditTemplateId(id);
    setEditOpen(true);
  };

  const openUse = (id: string) => {
    setUseTemplateId(id);
    setUseOpen(true);
  };

  const openDelete = (template: TemplateListItem) => {
    setDeleteTarget({ id: template.id, name: template.name });
    setDeleteOpen(true);
  };

  // Render ------------------------------------------------------------------

  const templateList: TemplateListItem[] = templates ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Prompt Library</h1>
          <p className="text-muted-foreground">
            Browse, create, and manage your prompt templates.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          New Prompt
        </Button>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>
        <Select value={selectedCategory} onValueChange={setSelectedCategory}>
          <SelectTrigger className="w-full sm:w-[200px]">
            <SelectValue placeholder="All Categories" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Categories</SelectItem>
            {(categories ?? []).map((cat) => (
              <SelectItem key={cat.name} value={cat.name}>
                {cat.name} ({cat.count})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Template grid */}
      {templatesLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader>
                <div className="h-5 w-5 rounded bg-muted" />
                <div className="h-4 w-3/4 rounded bg-muted mt-2" />
                <div className="h-3 w-full rounded bg-muted mt-2" />
              </CardHeader>
              <CardContent>
                <div className="h-3 w-1/2 rounded bg-muted" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : templateList.length === 0 ? (
        /* Empty state */
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16 text-center">
            <div className="p-4 rounded-full bg-muted/50 mb-4">
              <FileText className="h-10 w-10 text-muted-foreground opacity-50" />
            </div>
            <h3 className="text-lg font-semibold mb-1">No templates found</h3>
            <p className="text-sm text-muted-foreground max-w-sm mb-6">
              {searchQuery || selectedCategory !== "all"
                ? "Try adjusting your search or category filter."
                : "Get started by creating your first prompt template. Templates help you reuse prompts and share them with your team."}
            </p>
            {!searchQuery && selectedCategory === "all" && (
              <Button onClick={() => setCreateOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Create Your First Template
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {templateList.map((template) => (
            <Card
              key={template.id}
              className="flex flex-col hover:shadow-md transition-shadow"
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="p-1.5 rounded-md bg-primary/10 text-primary shrink-0">
                      {getCategoryIcon(template.category)}
                    </div>
                    <CardTitle className="text-base leading-tight truncate">
                      {template.name}
                    </CardTitle>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 shrink-0"
                      >
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => openUse(template.id)}>
                        <Eye className="h-4 w-4 mr-2" />
                        View &amp; Use
                      </DropdownMenuItem>
                      {template.is_owner && (
                        <DropdownMenuItem onClick={() => openEdit(template.id)}>
                          <Edit className="h-4 w-4 mr-2" />
                          Edit
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuItem
                        onClick={() => handleDuplicate(template)}
                      >
                        <Copy className="h-4 w-4 mr-2" />
                        Duplicate
                      </DropdownMenuItem>
                      {template.is_owner && (
                        <DropdownMenuItem
                          onClick={() => openDelete(template)}
                          className="text-destructive focus:text-destructive"
                        >
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                {template.description && (
                  <CardDescription className="line-clamp-2 mt-1">
                    {template.description}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="mt-auto pt-0">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 flex-wrap">
                    <Badge variant="secondary" className="text-xs">
                      {template.category}
                    </Badge>
                    {template.is_public && (
                      <Badge variant="outline" className="text-xs">
                        Public
                      </Badge>
                    )}
                    {template.is_system && (
                      <Badge variant="outline" className="text-xs border-primary/30 text-primary">
                        System
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                    <Star className="h-3 w-3" />
                    <span>{template.use_count} uses</span>
                  </div>
                </div>
                <div className="mt-3">
                  <Button
                    size="sm"
                    variant="outline"
                    className="w-full"
                    onClick={() => openUse(template.id)}
                  >
                    <Wand2 className="h-3.5 w-3.5 mr-1.5" />
                    Use Template
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Dialogs */}

      {/* Create dialog */}
      <TemplateFormDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        title="Create Template"
        description="Create a new prompt template for your library."
        initial={EMPTY_FORM}
        onSubmit={handleCreate}
        isSubmitting={createMutation.isPending}
      />

      {/* Edit dialog */}
      <TemplateFormDialog
        open={editOpen}
        onOpenChange={(open) => {
          setEditOpen(open);
          if (!open) setEditTemplateId(null);
        }}
        title="Edit Template"
        description="Update your prompt template."
        initial={editFormInitial}
        onSubmit={handleUpdate}
        isSubmitting={updateMutation.isPending}
      />

      {/* Use dialog */}
      <UseTemplateDialog
        open={useOpen}
        onOpenChange={setUseOpen}
        templateId={useTemplateId}
      />

      {/* Delete confirmation */}
      <DeleteConfirmDialog
        open={deleteOpen}
        onOpenChange={(open) => {
          setDeleteOpen(open);
          if (!open) setDeleteTarget(null);
        }}
        templateName={deleteTarget?.name ?? ""}
        onConfirm={handleDelete}
        isDeleting={deleteMutation.isPending}
      />
    </div>
  );
}
