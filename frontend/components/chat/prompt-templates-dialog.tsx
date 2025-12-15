"use client";

import * as React from "react";
import { FileText, Plus, Search, Loader2, Trash2, Copy } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  usePromptTemplates,
  usePromptTemplate,
  useCreatePromptTemplate,
  useApplyPromptTemplate,
  useDeletePromptTemplate,
} from "@/lib/api/hooks";
import type { TemplateListItem } from "@/lib/api/client";

interface PromptTemplatesDialogProps {
  onApply: (text: string) => void;
  trigger?: React.ReactNode;
}

export function PromptTemplatesDialog({
  onApply,
  trigger,
}: PromptTemplatesDialogProps) {
  const [open, setOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const [showCreate, setShowCreate] = React.useState(false);
  const [selectedTemplateId, setSelectedTemplateId] = React.useState<string | null>(null);
  const [variables, setVariables] = React.useState<Record<string, string>>({});

  // Form state for creating new template
  const [newName, setNewName] = React.useState("");
  const [newPrompt, setNewPrompt] = React.useState("");
  const [newCategory, setNewCategory] = React.useState("");

  const { data: templates, isLoading } = usePromptTemplates();
  const { data: selectedTemplate, isLoading: isLoadingTemplate } = usePromptTemplate(
    selectedTemplateId || "",
    { enabled: !!selectedTemplateId }
  );
  const createMutation = useCreatePromptTemplate();
  const applyMutation = useApplyPromptTemplate();
  const deleteMutation = useDeletePromptTemplate();

  // Filter templates based on search
  const filteredTemplates = React.useMemo(() => {
    if (!templates) return [];
    if (!search) return templates;
    const lower = search.toLowerCase();
    return templates.filter(
      (t) =>
        t.name.toLowerCase().includes(lower) ||
        t.category?.toLowerCase().includes(lower) ||
        t.description?.toLowerCase().includes(lower)
    );
  }, [templates, search]);

  // Group templates by category
  const groupedTemplates = React.useMemo(() => {
    const groups: Record<string, TemplateListItem[]> = {};
    for (const template of filteredTemplates) {
      const category = template.category || "Uncategorized";
      if (!groups[category]) {
        groups[category] = [];
      }
      groups[category].push(template);
    }
    return groups;
  }, [filteredTemplates]);

  // Initialize variables when template loads
  React.useEffect(() => {
    if (selectedTemplate?.variables) {
      const vars: Record<string, string> = {};
      for (const v of selectedTemplate.variables) {
        vars[v.name] = v.default || "";
      }
      setVariables(vars);
    }
  }, [selectedTemplate]);

  const handleSelectTemplate = (templateId: string) => {
    setSelectedTemplateId(templateId);
    setVariables({});
  };

  const handleApplyTemplate = async () => {
    if (!selectedTemplateId) return;

    try {
      const result = await applyMutation.mutateAsync({
        templateId: selectedTemplateId,
        variables,
      });
      onApply(result.prompt_text);
      setOpen(false);
      setSelectedTemplateId(null);
      setVariables({});
    } catch (error) {
      console.error("Failed to apply template:", error);
    }
  };

  const handleCreateTemplate = async () => {
    if (!newName.trim() || !newPrompt.trim()) return;

    try {
      await createMutation.mutateAsync({
        name: newName.trim(),
        prompt_text: newPrompt.trim(),
        category: newCategory.trim() || undefined,
        is_public: false,
      });
      setNewName("");
      setNewPrompt("");
      setNewCategory("");
      setShowCreate(false);
    } catch (error) {
      console.error("Failed to create template:", error);
    }
  };

  const handleDeleteTemplate = async (templateId: string) => {
    try {
      await deleteMutation.mutateAsync(templateId);
      if (selectedTemplateId === templateId) {
        setSelectedTemplateId(null);
      }
    } catch (error) {
      console.error("Failed to delete template:", error);
    }
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger || (
          <Button variant="ghost" size="icon" title="Prompt templates">
            <FileText className="h-5 w-5" />
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="max-w-3xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Prompt Templates</DialogTitle>
          <DialogDescription>
            Select a template to use or create your own. Use {"{{variable}}"} syntax for dynamic content.
          </DialogDescription>
        </DialogHeader>

        {showCreate ? (
          // Create new template form
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="template-name">Name</Label>
              <Input
                id="template-name"
                placeholder="My Template"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="template-category">Category (optional)</Label>
              <Input
                id="template-category"
                placeholder="e.g., Analysis, Research, Writing"
                value={newCategory}
                onChange={(e) => setNewCategory(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="template-prompt">Prompt Template</Label>
              <Textarea
                id="template-prompt"
                placeholder="Write your prompt here. Use {{variable_name}} for dynamic parts."
                className="min-h-[150px] font-mono text-sm"
                value={newPrompt}
                onChange={(e) => setNewPrompt(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                Example: &quot;Summarize the following document about {"{{topic}}"}: {"{{content}}"}&quot;
              </p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setShowCreate(false)}>
                Cancel
              </Button>
              <Button
                onClick={handleCreateTemplate}
                disabled={!newName.trim() || !newPrompt.trim() || createMutation.isPending}
              >
                {createMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                Create Template
              </Button>
            </DialogFooter>
          </div>
        ) : selectedTemplateId ? (
          // Apply template with variables
          <div className="space-y-4">
            {isLoadingTemplate ? (
              <div className="flex items-center justify-center h-32">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : selectedTemplate ? (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium">{selectedTemplate.name}</h3>
                    {selectedTemplate.category && (
                      <Badge variant="secondary" className="mt-1">
                        {selectedTemplate.category}
                      </Badge>
                    )}
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => setSelectedTemplateId(null)}>
                    Back to list
                  </Button>
                </div>

                <div className="rounded-md bg-muted p-3">
                  <p className="text-sm font-mono whitespace-pre-wrap">{selectedTemplate.prompt_text}</p>
                </div>

                {selectedTemplate.variables && selectedTemplate.variables.length > 0 && (
                  <div className="space-y-3">
                    <Label>Fill in variables:</Label>
                    {selectedTemplate.variables.map((v) => (
                      <div key={v.name} className="space-y-1">
                        <Label htmlFor={`var-${v.name}`} className="text-sm font-normal">
                          {v.name}
                          {v.description && (
                            <span className="text-muted-foreground ml-2">- {v.description}</span>
                          )}
                        </Label>
                        <Input
                          id={`var-${v.name}`}
                          placeholder={v.default || `Enter ${v.name}`}
                          value={variables[v.name] || ""}
                          onChange={(e) =>
                            setVariables((prev) => ({ ...prev, [v.name]: e.target.value }))
                          }
                        />
                      </div>
                    ))}
                  </div>
                )}

                <DialogFooter>
                  <Button variant="outline" onClick={() => setSelectedTemplateId(null)}>
                    Cancel
                  </Button>
                  <Button onClick={handleApplyTemplate} disabled={applyMutation.isPending}>
                    {applyMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Apply Template
                  </Button>
                </DialogFooter>
              </>
            ) : (
              <div className="text-center text-muted-foreground py-8">
                Template not found
              </div>
            )}
          </div>
        ) : (
          // Template list
          <div className="space-y-4">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search templates..."
                  className="pl-9"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
              <Button onClick={() => setShowCreate(true)}>
                <Plus className="mr-2 h-4 w-4" />
                New
              </Button>
            </div>

            <ScrollArea className="h-[400px]">
              {isLoading ? (
                <div className="flex items-center justify-center h-32">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : Object.keys(groupedTemplates).length === 0 ? (
                <div className="flex flex-col items-center justify-center h-32 text-center">
                  <FileText className="h-8 w-8 text-muted-foreground mb-2" />
                  <p className="text-muted-foreground">No templates found</p>
                  <Button
                    variant="link"
                    className="mt-2"
                    onClick={() => setShowCreate(true)}
                  >
                    Create your first template
                  </Button>
                </div>
              ) : (
                <div className="space-y-4">
                  {Object.entries(groupedTemplates).map(([category, categoryTemplates]) => (
                    <div key={category}>
                      <h4 className="text-sm font-medium text-muted-foreground mb-2">
                        {category}
                      </h4>
                      <div className="space-y-2">
                        {categoryTemplates.map((template) => (
                          <div
                            key={template.id}
                            className={cn(
                              "flex items-start justify-between p-3 rounded-lg border cursor-pointer hover:bg-accent transition-colors",
                              "group"
                            )}
                            onClick={() => handleSelectTemplate(template.id)}
                          >
                            <div className="flex-1 min-w-0">
                              <p className="font-medium truncate">{template.name}</p>
                              {template.description && (
                                <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
                                  {template.description}
                                </p>
                              )}
                              <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
                                {template.use_count > 0 && (
                                  <span>Used {template.use_count} times</span>
                                )}
                                {template.is_public && (
                                  <Badge variant="outline" className="text-xs">
                                    Public
                                  </Badge>
                                )}
                              </div>
                            </div>
                            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-destructive"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteTemplate(template.id);
                                }}
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
