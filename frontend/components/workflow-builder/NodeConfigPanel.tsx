"use client";

import { useState } from "react";
import { Trash2, Plus, X, ChevronDown, ChevronRight, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import type { NodeTypeInfo } from "@/lib/api";
import type { WorkflowNodeData, WorkflowNodeType } from "./WorkflowNode";

interface NodeConfigPanelProps {
  node: WorkflowNodeType;
  nodeTypes: NodeTypeInfo[];
  onUpdate: (updates: Partial<WorkflowNodeData>) => void;
  onDelete: () => void;
}

// Helper component for field labels with tooltips
function FieldLabel({ label, tooltip, required }: { label: string; tooltip?: string; required?: boolean }) {
  return (
    <div className="flex items-center gap-1">
      <Label>{label}</Label>
      {required && <span className="text-red-500">*</span>}
      {tooltip && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger>
              <Info className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p className="text-xs">{tooltip}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}

// Key-value pair editor for headers, params, etc.
function KeyValueEditor({
  value,
  onChange,
  keyPlaceholder = "Key",
  valuePlaceholder = "Value",
}: {
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
  keyPlaceholder?: string;
  valuePlaceholder?: string;
}) {
  const pairs = Object.entries(value || {});

  const addPair = () => {
    onChange({ ...value, "": "" });
  };

  const updatePair = (oldKey: string, newKey: string, newValue: string) => {
    const updated = { ...value };
    if (oldKey !== newKey) {
      delete updated[oldKey];
    }
    updated[newKey] = newValue;
    onChange(updated);
  };

  const removePair = (key: string) => {
    const updated = { ...value };
    delete updated[key];
    onChange(updated);
  };

  return (
    <div className="space-y-2">
      {pairs.map(([key, val], index) => (
        <div key={index} className="flex gap-2">
          <Input
            value={key}
            onChange={(e) => updatePair(key, e.target.value, val)}
            placeholder={keyPlaceholder}
            className="flex-1"
          />
          <Input
            value={val}
            onChange={(e) => updatePair(key, key, e.target.value)}
            placeholder={valuePlaceholder}
            className="flex-1"
          />
          <Button variant="ghost" size="icon" onClick={() => removePair(key)}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button variant="outline" size="sm" onClick={addPair} className="w-full">
        <Plus className="h-4 w-4 mr-1" /> Add
      </Button>
    </div>
  );
}

// Collapsible section component
function ConfigSection({
  title,
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <CollapsibleTrigger className="flex items-center gap-2 w-full py-2 text-sm font-medium hover:text-primary">
        {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-4 pt-2">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function NodeConfigPanel({
  node,
  nodeTypes,
  onUpdate,
  onDelete,
}: NodeConfigPanelProps) {
  const normalizedType = node.data?.nodeType?.toUpperCase() || "";
  const nodeTypeInfo = nodeTypes.find((nt) => nt.type.toUpperCase() === normalizedType);
  const isStartOrEnd = ["START", "END"].includes(normalizedType);

  const updateConfig = (key: string, value: unknown) => {
    onUpdate({
      config: {
        ...(node.data?.config || {}),
        [key]: value,
      },
    });
  };

  const getConfig = (key: string, defaultValue: string = ""): string => {
    return (node.data?.config?.[key] as string) ?? defaultValue;
  };

  const getConfigNumber = (key: string, defaultValue: number = 0): number => {
    return (node.data?.config?.[key] as number) ?? defaultValue;
  };

  const getConfigBool = (key: string, defaultValue: boolean = false): boolean => {
    return (node.data?.config?.[key] as boolean) ?? defaultValue;
  };

  const getConfigObject = <T extends Record<string, unknown>>(key: string, defaultValue: T): T => {
    return (node.data?.config?.[key] as T) ?? defaultValue;
  };

  const getConfigArray = <T,>(key: string, defaultValue: T[]): T[] => {
    return (node.data?.config?.[key] as T[]) ?? defaultValue;
  };

  return (
    <div className="py-4 space-y-6">
      {/* Basic Info */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="capitalize">
            {normalizedType.toLowerCase()}
          </Badge>
        </div>
        <div className="space-y-2">
          <FieldLabel label="Name" required />
          <Input
            value={node.data?.label || ""}
            onChange={(e) => onUpdate({ label: e.target.value })}
            placeholder="Node name"
          />
        </div>
        <div className="space-y-2">
          <FieldLabel label="Description" tooltip="Optional description for documentation" />
          <Textarea
            value={node.data?.description || ""}
            onChange={(e) => onUpdate({ description: e.target.value })}
            placeholder="What does this node do?"
            rows={2}
          />
        </div>
      </div>

      <Separator />

      {/* START Node Configuration */}
      {normalizedType === "START" && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium">Input Schema</h4>
          <p className="text-xs text-muted-foreground">
            Define the expected input data structure for this workflow.
          </p>
          <div className="space-y-2">
            <FieldLabel label="Input Variables" tooltip="Variables available in {{input.varName}} syntax" />
            <Textarea
              value={getConfig("input_schema", "")}
              onChange={(e) => updateConfig("input_schema", e.target.value)}
              placeholder={`{
  "document_id": "string",
  "options": {
    "format": "pdf | docx"
  }
}`}
              rows={6}
              className="font-mono text-sm"
            />
          </div>
        </div>
      )}

      {/* END Node Configuration */}
      {normalizedType === "END" && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium">Output Configuration</h4>
          <div className="space-y-2">
            <FieldLabel label="Output Path" tooltip="JSONPath to the data to return" />
            <Input
              value={getConfig("output_path", "$.result")}
              onChange={(e) => updateConfig("output_path", e.target.value)}
              placeholder="$.result"
              className="font-mono"
            />
          </div>
          <div className="flex items-center space-x-2">
            <Switch
              id="include_metadata"
              checked={getConfigBool("include_metadata", false)}
              onCheckedChange={(checked) => updateConfig("include_metadata", checked)}
            />
            <Label htmlFor="include_metadata">Include execution metadata</Label>
          </div>
        </div>
      )}

      {/* ACTION Node - Enhanced */}
      {normalizedType === "ACTION" && (
        <div className="space-y-4">
          <ConfigSection title="Action Settings">
            <div className="space-y-2">
              <FieldLabel label="Action Type" required />
              <Select
                value={getConfig("action_type", "")}
                onValueChange={(value) => updateConfig("action_type", value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select action type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="create_document">Create Document</SelectItem>
                  <SelectItem value="update_document">Update Document</SelectItem>
                  <SelectItem value="delete_document">Delete Document</SelectItem>
                  <SelectItem value="generate_pdf">Generate PDF</SelectItem>
                  <SelectItem value="generate_docx">Generate DOCX</SelectItem>
                  <SelectItem value="generate_pptx">Generate PPTX</SelectItem>
                  <SelectItem value="send_email">Send Email</SelectItem>
                  <SelectItem value="run_query">Run RAG Query</SelectItem>
                  <SelectItem value="embed_text">Embed Text</SelectItem>
                  <SelectItem value="transform_data">Transform Data</SelectItem>
                  <SelectItem value="set_variable">Set Variable</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Document Actions */}
            {["create_document", "update_document", "generate_pdf", "generate_docx", "generate_pptx"].includes(getConfig("action_type", "")) && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Template" tooltip="Template ID or path" />
                  <Input
                    value={getConfig("template_id", "")}
                    onChange={(e) => updateConfig("template_id", e.target.value)}
                    placeholder="{{input.template_id}}"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Data Mapping" tooltip="Map input data to template variables" />
                  <Textarea
                    value={getConfig("data_mapping", "")}
                    onChange={(e) => updateConfig("data_mapping", e.target.value)}
                    placeholder={`{
  "title": "{{input.title}}",
  "content": "{{nodes.agent_1.output}}"
}`}
                    rows={5}
                    className="font-mono text-sm"
                  />
                </div>
              </>
            )}

            {/* Email Action */}
            {getConfig("action_type", "") === "send_email" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="To" required />
                  <Input
                    value={getConfig("email_to", "")}
                    onChange={(e) => updateConfig("email_to", e.target.value)}
                    placeholder="{{input.recipient}}"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Subject" required />
                  <Input
                    value={getConfig("email_subject", "")}
                    onChange={(e) => updateConfig("email_subject", e.target.value)}
                    placeholder="Workflow Result: {{workflow.name}}"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Body Template" />
                  <Textarea
                    value={getConfig("email_body", "")}
                    onChange={(e) => updateConfig("email_body", e.target.value)}
                    placeholder="Dear {{input.name}},\n\nYour request has been processed..."
                    rows={5}
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="email_html"
                    checked={getConfigBool("email_html", false)}
                    onCheckedChange={(checked) => updateConfig("email_html", checked)}
                  />
                  <Label htmlFor="email_html">HTML Email</Label>
                </div>
              </>
            )}

            {/* RAG Query */}
            {getConfig("action_type", "") === "run_query" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Query" required />
                  <Textarea
                    value={getConfig("query", "")}
                    onChange={(e) => updateConfig("query", e.target.value)}
                    placeholder="{{input.question}}"
                    rows={3}
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Top K Results" />
                  <Slider
                    value={[getConfigNumber("top_k", 5)]}
                    onValueChange={([value]) => updateConfig("top_k", value)}
                    min={1}
                    max={20}
                    step={1}
                  />
                  <p className="text-xs text-muted-foreground text-right">{getConfigNumber("top_k", 5)} results</p>
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Filter by Folder" />
                  <Input
                    value={getConfig("folder_filter", "")}
                    onChange={(e) => updateConfig("folder_filter", e.target.value)}
                    placeholder="Leave empty for all documents"
                  />
                </div>
              </>
            )}

            {/* Transform Data */}
            {getConfig("action_type", "") === "transform_data" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Transform Type" />
                  <Select
                    value={getConfig("transform_type", "jq")}
                    onValueChange={(value) => updateConfig("transform_type", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="jq">JQ Expression</SelectItem>
                      <SelectItem value="jsonata">JSONata</SelectItem>
                      <SelectItem value="template">Template</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Expression" required />
                  <Textarea
                    value={getConfig("transform_expression", "")}
                    onChange={(e) => updateConfig("transform_expression", e.target.value)}
                    placeholder=".items | map({id: .id, name: .title})"
                    rows={4}
                    className="font-mono text-sm"
                  />
                </div>
              </>
            )}
          </ConfigSection>

          <ConfigSection title="Error Handling" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="On Error" />
              <Select
                value={getConfig("on_error", "fail")}
                onValueChange={(value) => updateConfig("on_error", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fail">Stop Workflow</SelectItem>
                  <SelectItem value="continue">Continue</SelectItem>
                  <SelectItem value="retry">Retry</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {getConfig("on_error", "fail") === "retry" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Max Retries" />
                  <Input
                    type="number"
                    value={getConfigNumber("max_retries", 3)}
                    onChange={(e) => updateConfig("max_retries", parseInt(e.target.value) || 3)}
                    min={1}
                    max={10}
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Retry Delay (seconds)" />
                  <Input
                    type="number"
                    value={getConfigNumber("retry_delay", 5)}
                    onChange={(e) => updateConfig("retry_delay", parseInt(e.target.value) || 5)}
                    min={1}
                  />
                </div>
              </>
            )}
          </ConfigSection>
        </div>
      )}

      {/* CONDITION Node - Enhanced */}
      {normalizedType === "CONDITION" && (
        <div className="space-y-4">
          <ConfigSection title="Condition Settings">
            <div className="space-y-2">
              <FieldLabel label="Condition Type" required />
              <Select
                value={getConfig("condition_type", "expression")}
                onValueChange={(value) => updateConfig("condition_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="expression">JavaScript Expression</SelectItem>
                  <SelectItem value="compare">Compare Values</SelectItem>
                  <SelectItem value="exists">Value Exists</SelectItem>
                  <SelectItem value="type_check">Type Check</SelectItem>
                  <SelectItem value="regex">Regex Match</SelectItem>
                  <SelectItem value="all">All Conditions (AND)</SelectItem>
                  <SelectItem value="any">Any Condition (OR)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {getConfig("condition_type", "expression") === "expression" && (
              <div className="space-y-2">
                <FieldLabel label="Expression" required tooltip="JavaScript expression that returns true/false" />
                <Textarea
                  value={getConfig("expression", "")}
                  onChange={(e) => updateConfig("expression", e.target.value)}
                  placeholder={`// Examples:
input.status === "approved"
input.score > 0.8
nodes.agent.output.includes("success")`}
                  rows={5}
                  className="font-mono text-sm"
                />
              </div>
            )}

            {getConfig("condition_type", "expression") === "compare" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Left Value" required />
                  <Input
                    value={getConfig("left_value", "")}
                    onChange={(e) => updateConfig("left_value", e.target.value)}
                    placeholder="{{input.status}}"
                    className="font-mono"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Operator" required />
                  <Select
                    value={getConfig("operator", "equals")}
                    onValueChange={(value) => updateConfig("operator", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="equals">Equals (==)</SelectItem>
                      <SelectItem value="not_equals">Not Equals (!=)</SelectItem>
                      <SelectItem value="greater">Greater Than (&gt;)</SelectItem>
                      <SelectItem value="greater_eq">Greater or Equal (&gt;=)</SelectItem>
                      <SelectItem value="less">Less Than (&lt;)</SelectItem>
                      <SelectItem value="less_eq">Less or Equal (&lt;=)</SelectItem>
                      <SelectItem value="contains">Contains</SelectItem>
                      <SelectItem value="starts_with">Starts With</SelectItem>
                      <SelectItem value="ends_with">Ends With</SelectItem>
                      <SelectItem value="in">In Array</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Right Value" required />
                  <Input
                    value={getConfig("right_value", "")}
                    onChange={(e) => updateConfig("right_value", e.target.value)}
                    placeholder="approved"
                    className="font-mono"
                  />
                </div>
              </>
            )}

            {getConfig("condition_type", "expression") === "regex" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Input Value" required />
                  <Input
                    value={getConfig("regex_input", "")}
                    onChange={(e) => updateConfig("regex_input", e.target.value)}
                    placeholder="{{input.email}}"
                    className="font-mono"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Regex Pattern" required />
                  <Input
                    value={getConfig("regex_pattern", "")}
                    onChange={(e) => updateConfig("regex_pattern", e.target.value)}
                    placeholder="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
                    className="font-mono"
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <Switch
                    id="regex_case_insensitive"
                    checked={getConfigBool("regex_case_insensitive", false)}
                    onCheckedChange={(checked) => updateConfig("regex_case_insensitive", checked)}
                  />
                  <Label htmlFor="regex_case_insensitive">Case Insensitive</Label>
                </div>
              </>
            )}
          </ConfigSection>

          <ConfigSection title="Branch Labels" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="True Branch Label" />
              <Input
                value={getConfig("true_label", "Yes")}
                onChange={(e) => updateConfig("true_label", e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <FieldLabel label="False Branch Label" />
              <Input
                value={getConfig("false_label", "No")}
                onChange={(e) => updateConfig("false_label", e.target.value)}
              />
            </div>
          </ConfigSection>
        </div>
      )}

      {/* LOOP Node - Enhanced */}
      {normalizedType === "LOOP" && (
        <div className="space-y-4">
          <ConfigSection title="Loop Settings">
            <div className="space-y-2">
              <FieldLabel label="Loop Type" required />
              <Select
                value={getConfig("loop_type", "for_each")}
                onValueChange={(value) => updateConfig("loop_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="for_each">For Each Item</SelectItem>
                  <SelectItem value="while">While Condition</SelectItem>
                  <SelectItem value="count">Fixed Count</SelectItem>
                  <SelectItem value="batch">Batch Processing</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {getConfig("loop_type", "for_each") === "for_each" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Items Array" required tooltip="Path to the array to iterate" />
                  <Input
                    value={getConfig("items_source", "")}
                    onChange={(e) => updateConfig("items_source", e.target.value)}
                    placeholder="{{input.documents}}"
                    className="font-mono"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Item Variable Name" tooltip="Access current item as {{loop.item}}" />
                  <Input
                    value={getConfig("item_var", "item")}
                    onChange={(e) => updateConfig("item_var", e.target.value)}
                    placeholder="item"
                    className="font-mono"
                  />
                </div>
              </>
            )}

            {getConfig("loop_type", "for_each") === "while" && (
              <div className="space-y-2">
                <FieldLabel label="Condition" required tooltip="Loop continues while true" />
                <Textarea
                  value={getConfig("while_condition", "")}
                  onChange={(e) => updateConfig("while_condition", e.target.value)}
                  placeholder="loop.index < 10 && !loop.result.done"
                  rows={3}
                  className="font-mono text-sm"
                />
              </div>
            )}

            {getConfig("loop_type", "for_each") === "count" && (
              <div className="space-y-2">
                <FieldLabel label="Iterations" required />
                <Input
                  type="number"
                  value={getConfigNumber("count", 5)}
                  onChange={(e) => updateConfig("count", parseInt(e.target.value) || 5)}
                  min={1}
                />
              </div>
            )}

            {getConfig("loop_type", "for_each") === "batch" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Items Array" required />
                  <Input
                    value={getConfig("items_source", "")}
                    onChange={(e) => updateConfig("items_source", e.target.value)}
                    placeholder="{{input.documents}}"
                    className="font-mono"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Batch Size" />
                  <Input
                    type="number"
                    value={getConfigNumber("batch_size", 10)}
                    onChange={(e) => updateConfig("batch_size", parseInt(e.target.value) || 10)}
                    min={1}
                  />
                </div>
              </>
            )}
          </ConfigSection>

          <ConfigSection title="Limits & Safety" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Max Iterations" tooltip="Prevent infinite loops" />
              <Input
                type="number"
                value={getConfigNumber("max_iterations", 100)}
                onChange={(e) => updateConfig("max_iterations", parseInt(e.target.value) || 100)}
                min={1}
                max={10000}
              />
            </div>
            <div className="space-y-2">
              <FieldLabel label="Delay Between Iterations (ms)" />
              <Input
                type="number"
                value={getConfigNumber("delay_ms", 0)}
                onChange={(e) => updateConfig("delay_ms", parseInt(e.target.value) || 0)}
                min={0}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="parallel"
                checked={getConfigBool("parallel", false)}
                onCheckedChange={(checked) => updateConfig("parallel", checked)}
              />
              <Label htmlFor="parallel">Run in Parallel</Label>
            </div>
            {getConfigBool("parallel", false) && (
              <div className="space-y-2">
                <FieldLabel label="Concurrency Limit" />
                <Input
                  type="number"
                  value={getConfigNumber("concurrency", 5)}
                  onChange={(e) => updateConfig("concurrency", parseInt(e.target.value) || 5)}
                  min={1}
                  max={50}
                />
              </div>
            )}
          </ConfigSection>
        </div>
      )}

      {/* CODE Node - Enhanced */}
      {normalizedType === "CODE" && (
        <div className="space-y-4">
          <ConfigSection title="Code Settings">
            <div className="space-y-2">
              <FieldLabel label="Language" required />
              <Select
                value={getConfig("language", "python")}
                onValueChange={(value) => updateConfig("language", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="python">Python</SelectItem>
                  <SelectItem value="javascript">JavaScript</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Execution Mode" />
              <Select
                value={getConfig("mode", "all_items")}
                onValueChange={(value) => updateConfig("mode", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all_items">Run Once for All Items</SelectItem>
                  <SelectItem value="each_item">Run for Each Item</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Code" required />
              <Textarea
                value={getConfig("code", "")}
                onChange={(e) => updateConfig("code", e.target.value)}
                placeholder={getConfig("language", "python") === "python"
                  ? `# Available variables:
# - input: dict with input data
# - nodes: dict with outputs from previous nodes
# - context: workflow context

def main(input, nodes, context):
    result = input.get("data", [])
    # Transform data
    return {"processed": result}`
                  : `// Available variables:
// - input: object with input data
// - nodes: object with outputs from previous nodes
// - context: workflow context

function main(input, nodes, context) {
  const result = input.data || [];
  // Transform data
  return { processed: result };
}`}
                rows={15}
                className="font-mono text-sm"
              />
            </div>
          </ConfigSection>

          <ConfigSection title="Advanced" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Timeout (seconds)" />
              <Input
                type="number"
                value={getConfigNumber("timeout", 30)}
                onChange={(e) => updateConfig("timeout", parseInt(e.target.value) || 30)}
                min={1}
                max={300}
              />
            </div>
            <div className="space-y-2">
              <FieldLabel label="Memory Limit (MB)" />
              <Input
                type="number"
                value={getConfigNumber("memory_limit", 256)}
                onChange={(e) => updateConfig("memory_limit", parseInt(e.target.value) || 256)}
                min={64}
                max={2048}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="sandbox"
                checked={getConfigBool("sandbox", true)}
                onCheckedChange={(checked) => updateConfig("sandbox", checked)}
              />
              <Label htmlFor="sandbox">Run in Sandbox</Label>
            </div>
          </ConfigSection>
        </div>
      )}

      {/* DELAY Node - Enhanced */}
      {normalizedType === "DELAY" && (
        <div className="space-y-4">
          <ConfigSection title="Delay Settings">
            <div className="space-y-2">
              <FieldLabel label="Delay Type" />
              <Select
                value={getConfig("delay_type", "fixed")}
                onValueChange={(value) => updateConfig("delay_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixed">Fixed Duration</SelectItem>
                  <SelectItem value="until">Until Specific Time</SelectItem>
                  <SelectItem value="cron">Next Cron Occurrence</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {getConfig("delay_type", "fixed") === "fixed" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Duration" />
                  <div className="flex gap-2">
                    <Input
                      type="number"
                      value={getConfigNumber("delay_value", 0)}
                      onChange={(e) => updateConfig("delay_value", parseInt(e.target.value) || 0)}
                      min={0}
                      className="flex-1"
                    />
                    <Select
                      value={getConfig("delay_unit", "seconds")}
                      onValueChange={(value) => updateConfig("delay_unit", value)}
                    >
                      <SelectTrigger className="w-32">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="seconds">Seconds</SelectItem>
                        <SelectItem value="minutes">Minutes</SelectItem>
                        <SelectItem value="hours">Hours</SelectItem>
                        <SelectItem value="days">Days</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </>
            )}

            {getConfig("delay_type", "fixed") === "until" && (
              <div className="space-y-2">
                <FieldLabel label="Wait Until" tooltip="ISO 8601 datetime or expression" />
                <Input
                  value={getConfig("until_time", "")}
                  onChange={(e) => updateConfig("until_time", e.target.value)}
                  placeholder="2024-12-31T23:59:59Z or {{input.deadline}}"
                  className="font-mono"
                />
              </div>
            )}

            {getConfig("delay_type", "fixed") === "cron" && (
              <div className="space-y-2">
                <FieldLabel label="Cron Expression" />
                <Input
                  value={getConfig("cron", "")}
                  onChange={(e) => updateConfig("cron", e.target.value)}
                  placeholder="0 9 * * 1-5"
                  className="font-mono"
                />
                <p className="text-xs text-muted-foreground">
                  Wait until next occurrence (e.g., 0 9 * * 1-5 = 9 AM weekdays)
                </p>
              </div>
            )}
          </ConfigSection>
        </div>
      )}

      {/* HTTP Node - Enhanced */}
      {normalizedType === "HTTP" && (
        <div className="space-y-4">
          <ConfigSection title="Request Settings">
            <div className="space-y-2">
              <FieldLabel label="Method" required />
              <Select
                value={getConfig("method", "GET")}
                onValueChange={(value) => updateConfig("method", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="GET">GET</SelectItem>
                  <SelectItem value="POST">POST</SelectItem>
                  <SelectItem value="PUT">PUT</SelectItem>
                  <SelectItem value="PATCH">PATCH</SelectItem>
                  <SelectItem value="DELETE">DELETE</SelectItem>
                  <SelectItem value="HEAD">HEAD</SelectItem>
                  <SelectItem value="OPTIONS">OPTIONS</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <FieldLabel label="URL" required />
              <Input
                value={getConfig("url", "")}
                onChange={(e) => updateConfig("url", e.target.value)}
                placeholder="https://api.example.com/v1/{{input.endpoint}}"
                className="font-mono"
              />
            </div>
          </ConfigSection>

          <ConfigSection title="Authentication" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Auth Type" />
              <Select
                value={getConfig("auth_type", "none")}
                onValueChange={(value) => updateConfig("auth_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="basic">Basic Auth</SelectItem>
                  <SelectItem value="bearer">Bearer Token</SelectItem>
                  <SelectItem value="api_key">API Key</SelectItem>
                  <SelectItem value="oauth2">OAuth2</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {getConfig("auth_type", "none") === "basic" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Username" />
                  <Input
                    value={getConfig("auth_username", "")}
                    onChange={(e) => updateConfig("auth_username", e.target.value)}
                    placeholder="{{secrets.api_username}}"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Password" />
                  <Input
                    type="password"
                    value={getConfig("auth_password", "")}
                    onChange={(e) => updateConfig("auth_password", e.target.value)}
                    placeholder="{{secrets.api_password}}"
                  />
                </div>
              </>
            )}

            {getConfig("auth_type", "none") === "bearer" && (
              <div className="space-y-2">
                <FieldLabel label="Token" />
                <Input
                  value={getConfig("auth_token", "")}
                  onChange={(e) => updateConfig("auth_token", e.target.value)}
                  placeholder="{{secrets.api_token}}"
                />
              </div>
            )}

            {getConfig("auth_type", "none") === "api_key" && (
              <>
                <div className="space-y-2">
                  <FieldLabel label="Key Name" />
                  <Input
                    value={getConfig("api_key_name", "")}
                    onChange={(e) => updateConfig("api_key_name", e.target.value)}
                    placeholder="X-API-Key"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Key Value" />
                  <Input
                    value={getConfig("api_key_value", "")}
                    onChange={(e) => updateConfig("api_key_value", e.target.value)}
                    placeholder="{{secrets.api_key}}"
                  />
                </div>
                <div className="space-y-2">
                  <FieldLabel label="Location" />
                  <Select
                    value={getConfig("api_key_location", "header")}
                    onValueChange={(value) => updateConfig("api_key_location", value)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="header">Header</SelectItem>
                      <SelectItem value="query">Query Parameter</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </>
            )}
          </ConfigSection>

          <ConfigSection title="Headers" defaultOpen={false}>
            <KeyValueEditor
              value={getConfigObject<Record<string, string>>("headers", {})}
              onChange={(val) => updateConfig("headers", val)}
              keyPlaceholder="Header name"
              valuePlaceholder="Header value"
            />
          </ConfigSection>

          <ConfigSection title="Query Parameters" defaultOpen={false}>
            <KeyValueEditor
              value={getConfigObject<Record<string, string>>("query_params", {})}
              onChange={(val) => updateConfig("query_params", val)}
              keyPlaceholder="Parameter"
              valuePlaceholder="Value"
            />
          </ConfigSection>

          {["POST", "PUT", "PATCH"].includes(getConfig("method", "GET")) && (
            <ConfigSection title="Request Body">
              <div className="space-y-2">
                <FieldLabel label="Content Type" />
                <Select
                  value={getConfig("content_type", "json")}
                  onValueChange={(value) => updateConfig("content_type", value)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="json">JSON</SelectItem>
                    <SelectItem value="form">Form Data</SelectItem>
                    <SelectItem value="multipart">Multipart Form</SelectItem>
                    <SelectItem value="raw">Raw Text</SelectItem>
                    <SelectItem value="xml">XML</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <FieldLabel label="Body" />
                <Textarea
                  value={getConfig("body", "")}
                  onChange={(e) => updateConfig("body", e.target.value)}
                  placeholder={`{
  "data": "{{input.data}}",
  "timestamp": "{{now}}"
}`}
                  rows={8}
                  className="font-mono text-sm"
                />
              </div>
            </ConfigSection>
          )}

          <ConfigSection title="Response Handling" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Response Type" />
              <Select
                value={getConfig("response_type", "json")}
                onValueChange={(value) => updateConfig("response_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="binary">Binary</SelectItem>
                  <SelectItem value="auto">Auto-detect</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <FieldLabel label="Extract Path" tooltip="JSONPath to extract from response" />
              <Input
                value={getConfig("extract_path", "")}
                onChange={(e) => updateConfig("extract_path", e.target.value)}
                placeholder="$.data.results"
                className="font-mono"
              />
            </div>
          </ConfigSection>

          <ConfigSection title="Retry & Timeout" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Timeout (seconds)" />
              <Input
                type="number"
                value={getConfigNumber("timeout", 30)}
                onChange={(e) => updateConfig("timeout", parseInt(e.target.value) || 30)}
                min={1}
                max={300}
              />
            </div>
            <div className="space-y-2">
              <FieldLabel label="Retry Count" />
              <Input
                type="number"
                value={getConfigNumber("retries", 0)}
                onChange={(e) => updateConfig("retries", parseInt(e.target.value) || 0)}
                min={0}
                max={10}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="follow_redirects"
                checked={getConfigBool("follow_redirects", true)}
                onCheckedChange={(checked) => updateConfig("follow_redirects", checked)}
              />
              <Label htmlFor="follow_redirects">Follow Redirects</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="ignore_ssl"
                checked={getConfigBool("ignore_ssl", false)}
                onCheckedChange={(checked) => updateConfig("ignore_ssl", checked)}
              />
              <Label htmlFor="ignore_ssl">Ignore SSL Errors</Label>
            </div>
          </ConfigSection>
        </div>
      )}

      {/* NOTIFICATION Node - Enhanced */}
      {normalizedType === "NOTIFICATION" && (
        <div className="space-y-4">
          <ConfigSection title="Notification Settings">
            <div className="space-y-2">
              <FieldLabel label="Channel" required />
              <Select
                value={getConfig("channel", "email")}
                onValueChange={(value) => updateConfig("channel", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="email">Email</SelectItem>
                  <SelectItem value="slack">Slack</SelectItem>
                  <SelectItem value="teams">Microsoft Teams</SelectItem>
                  <SelectItem value="discord">Discord</SelectItem>
                  <SelectItem value="webhook">Custom Webhook</SelectItem>
                  <SelectItem value="sms">SMS (Twilio)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Recipients" required />
              <Textarea
                value={getConfig("recipients", "")}
                onChange={(e) => updateConfig("recipients", e.target.value)}
                placeholder={getConfig("channel", "email") === "slack"
                  ? "#general, @user"
                  : "email@example.com, email2@example.com"}
                rows={2}
              />
            </div>

            <div className="space-y-2">
              <FieldLabel label="Subject / Title" />
              <Input
                value={getConfig("subject", "")}
                onChange={(e) => updateConfig("subject", e.target.value)}
                placeholder="Workflow {{workflow.name}} completed"
              />
            </div>

            <div className="space-y-2">
              <FieldLabel label="Message" required />
              <Textarea
                value={getConfig("message", "")}
                onChange={(e) => updateConfig("message", e.target.value)}
                placeholder={`Status: {{nodes.previous.status}}
Result: {{nodes.previous.output}}

View details: {{workflow.url}}`}
                rows={6}
              />
            </div>

            {getConfig("channel", "email") === "email" && (
              <div className="flex items-center space-x-2">
                <Switch
                  id="html_email"
                  checked={getConfigBool("html_email", false)}
                  onCheckedChange={(checked) => updateConfig("html_email", checked)}
                />
                <Label htmlFor="html_email">HTML Format</Label>
              </div>
            )}

            {["slack", "teams", "discord"].includes(getConfig("channel", "email")) && (
              <div className="space-y-2">
                <FieldLabel label="Webhook URL" tooltip="For private channels or custom integrations" />
                <Input
                  value={getConfig("webhook_url", "")}
                  onChange={(e) => updateConfig("webhook_url", e.target.value)}
                  placeholder="https://hooks.slack.com/services/..."
                />
              </div>
            )}
          </ConfigSection>

          <ConfigSection title="Attachments" defaultOpen={false}>
            <div className="flex items-center space-x-2">
              <Switch
                id="include_output"
                checked={getConfigBool("include_output", false)}
                onCheckedChange={(checked) => updateConfig("include_output", checked)}
              />
              <Label htmlFor="include_output">Include Workflow Output</Label>
            </div>
            <div className="space-y-2">
              <FieldLabel label="Attachment Path" tooltip="Path to file to attach" />
              <Input
                value={getConfig("attachment_path", "")}
                onChange={(e) => updateConfig("attachment_path", e.target.value)}
                placeholder="{{nodes.generate.output.file_path}}"
                className="font-mono"
              />
            </div>
          </ConfigSection>
        </div>
      )}

      {/* AGENT Node - Enhanced */}
      {normalizedType === "AGENT" && (
        <div className="space-y-4">
          <ConfigSection title="Agent Configuration">
            <div className="space-y-2">
              <FieldLabel label="Agent Type" required />
              <Select
                value={getConfig("agent_type", "default")}
                onValueChange={(value) => updateConfig("agent_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default RAG Agent</SelectItem>
                  <SelectItem value="researcher">Research Agent</SelectItem>
                  <SelectItem value="writer">Writing Agent</SelectItem>
                  <SelectItem value="coder">Code Agent</SelectItem>
                  <SelectItem value="analyst">Data Analyst</SelectItem>
                  <SelectItem value="custom">Custom Agent</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {getConfig("agent_type", "default") === "custom" && (
              <div className="space-y-2">
                <FieldLabel label="Agent ID" required />
                <Input
                  value={getConfig("agent_id", "")}
                  onChange={(e) => updateConfig("agent_id", e.target.value)}
                  placeholder="custom-agent-id"
                />
              </div>
            )}

            <div className="space-y-2">
              <FieldLabel label="Prompt / Task" required />
              <Textarea
                value={getConfig("prompt", "")}
                onChange={(e) => updateConfig("prompt", e.target.value)}
                placeholder={`Analyze the following document and extract key insights:

{{input.document}}

Focus on:
1. Main topics
2. Key findings
3. Recommendations`}
                rows={8}
              />
            </div>

            <div className="space-y-2">
              <FieldLabel label="System Instructions" tooltip="Optional context for the agent" />
              <Textarea
                value={getConfig("system_prompt", "")}
                onChange={(e) => updateConfig("system_prompt", e.target.value)}
                placeholder="You are an expert analyst..."
                rows={3}
              />
            </div>
          </ConfigSection>

          <ConfigSection title="Model Settings" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Model" />
              <Select
                value={getConfig("model", "default")}
                onValueChange={(value) => updateConfig("model", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default (from settings)</SelectItem>
                  <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                  <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                  <SelectItem value="claude-3-5-sonnet">Claude 3.5 Sonnet</SelectItem>
                  <SelectItem value="claude-3-opus">Claude 3 Opus</SelectItem>
                  <SelectItem value="gemini-pro">Gemini Pro</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Temperature" />
              <Slider
                value={[getConfigNumber("temperature", 0.7)]}
                onValueChange={([value]) => updateConfig("temperature", value)}
                min={0}
                max={2}
                step={0.1}
              />
              <p className="text-xs text-muted-foreground text-right">{getConfigNumber("temperature", 0.7)}</p>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Max Tokens" />
              <Input
                type="number"
                value={getConfigNumber("max_tokens", 4096)}
                onChange={(e) => updateConfig("max_tokens", parseInt(e.target.value) || 4096)}
                min={100}
                max={128000}
              />
            </div>
          </ConfigSection>

          <ConfigSection title="Context & Tools" defaultOpen={false}>
            <div className="flex items-center space-x-2">
              <Switch
                id="use_rag"
                checked={getConfigBool("use_rag", true)}
                onCheckedChange={(checked) => updateConfig("use_rag", checked)}
              />
              <Label htmlFor="use_rag">Use RAG (Document Search)</Label>
            </div>

            {getConfigBool("use_rag", true) && (
              <div className="space-y-2">
                <FieldLabel label="Document Filter" />
                <Input
                  value={getConfig("doc_filter", "")}
                  onChange={(e) => updateConfig("doc_filter", e.target.value)}
                  placeholder="folder:Marketing, tag:confidential"
                />
              </div>
            )}

            <div className="flex items-center space-x-2">
              <Switch
                id="use_web"
                checked={getConfigBool("use_web", false)}
                onCheckedChange={(checked) => updateConfig("use_web", checked)}
              />
              <Label htmlFor="use_web">Enable Web Search</Label>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="use_code"
                checked={getConfigBool("use_code", false)}
                onCheckedChange={(checked) => updateConfig("use_code", checked)}
              />
              <Label htmlFor="use_code">Enable Code Execution</Label>
            </div>
          </ConfigSection>

          <ConfigSection title="Execution" defaultOpen={false}>
            <div className="flex items-center space-x-2">
              <Switch
                id="wait_for_result"
                checked={getConfigBool("wait_for_result", true)}
                onCheckedChange={(checked) => updateConfig("wait_for_result", checked)}
              />
              <Label htmlFor="wait_for_result">Wait for Result</Label>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Timeout (seconds)" />
              <Input
                type="number"
                value={getConfigNumber("timeout", 120)}
                onChange={(e) => updateConfig("timeout", parseInt(e.target.value) || 120)}
                min={10}
                max={600}
              />
            </div>

            <div className="space-y-2">
              <FieldLabel label="Output Format" />
              <Select
                value={getConfig("output_format", "text")}
                onValueChange={(value) => updateConfig("output_format", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text">Plain Text</SelectItem>
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="markdown">Markdown</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </ConfigSection>
        </div>
      )}

      {/* HUMAN_APPROVAL Node - Enhanced */}
      {normalizedType === "HUMAN_APPROVAL" && (
        <div className="space-y-4">
          <ConfigSection title="Approval Settings">
            <div className="space-y-2">
              <FieldLabel label="Approvers" required tooltip="Comma-separated emails or user IDs" />
              <Textarea
                value={getConfig("approvers", "")}
                onChange={(e) => updateConfig("approvers", e.target.value)}
                placeholder="manager@example.com, admin@example.com"
                rows={2}
              />
            </div>

            <div className="space-y-2">
              <FieldLabel label="Approval Type" />
              <Select
                value={getConfig("approval_type", "any")}
                onValueChange={(value) => updateConfig("approval_type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="any">Any One Approver</SelectItem>
                  <SelectItem value="all">All Approvers</SelectItem>
                  <SelectItem value="majority">Majority</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <FieldLabel label="Title" />
              <Input
                value={getConfig("title", "")}
                onChange={(e) => updateConfig("title", e.target.value)}
                placeholder="Approval Required: {{workflow.name}}"
              />
            </div>

            <div className="space-y-2">
              <FieldLabel label="Message" required />
              <Textarea
                value={getConfig("message", "")}
                onChange={(e) => updateConfig("message", e.target.value)}
                placeholder={`Please review and approve the following:

Document: {{input.document_name}}
Action: {{input.action}}

Data: {{nodes.previous.output | json}}`}
                rows={6}
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="allow_comments"
                checked={getConfigBool("allow_comments", true)}
                onCheckedChange={(checked) => updateConfig("allow_comments", checked)}
              />
              <Label htmlFor="allow_comments">Allow Comments</Label>
            </div>
          </ConfigSection>

          <ConfigSection title="Timeout & Escalation" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Timeout" />
              <div className="flex gap-2">
                <Input
                  type="number"
                  value={getConfigNumber("timeout_value", 24)}
                  onChange={(e) => updateConfig("timeout_value", parseInt(e.target.value) || 24)}
                  min={1}
                  className="flex-1"
                />
                <Select
                  value={getConfig("timeout_unit", "hours")}
                  onValueChange={(value) => updateConfig("timeout_unit", value)}
                >
                  <SelectTrigger className="w-28">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="minutes">Minutes</SelectItem>
                    <SelectItem value="hours">Hours</SelectItem>
                    <SelectItem value="days">Days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <FieldLabel label="On Timeout" />
              <Select
                value={getConfig("on_timeout", "reject")}
                onValueChange={(value) => updateConfig("on_timeout", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="reject">Auto-Reject</SelectItem>
                  <SelectItem value="approve">Auto-Approve</SelectItem>
                  <SelectItem value="escalate">Escalate</SelectItem>
                  <SelectItem value="remind">Send Reminder</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {getConfig("on_timeout", "reject") === "escalate" && (
              <div className="space-y-2">
                <FieldLabel label="Escalation To" />
                <Input
                  value={getConfig("escalation_to", "")}
                  onChange={(e) => updateConfig("escalation_to", e.target.value)}
                  placeholder="supervisor@example.com"
                />
              </div>
            )}
          </ConfigSection>

          <ConfigSection title="Notification" defaultOpen={false}>
            <div className="space-y-2">
              <FieldLabel label="Notification Channels" />
              <div className="flex flex-wrap gap-2">
                {["email", "slack", "teams", "in_app"].map((channel) => (
                  <Button
                    key={channel}
                    variant={getConfigArray<string>("notify_channels", ["email"]).includes(channel) ? "default" : "outline"}
                    size="sm"
                    onClick={() => {
                      const current = getConfigArray<string>("notify_channels", ["email"]);
                      if (current.includes(channel)) {
                        updateConfig("notify_channels", current.filter((c: string) => c !== channel));
                      } else {
                        updateConfig("notify_channels", [...current, channel]);
                      }
                    }}
                  >
                    {channel.replace("_", " ")}
                  </Button>
                ))}
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="send_reminder"
                checked={getConfigBool("send_reminder", true)}
                onCheckedChange={(checked) => updateConfig("send_reminder", checked)}
              />
              <Label htmlFor="send_reminder">Send Reminders</Label>
            </div>

            {getConfigBool("send_reminder", true) && (
              <div className="space-y-2">
                <FieldLabel label="Reminder Interval (hours)" />
                <Input
                  type="number"
                  value={getConfigNumber("reminder_interval", 4)}
                  onChange={(e) => updateConfig("reminder_interval", parseInt(e.target.value) || 4)}
                  min={1}
                />
              </div>
            )}
          </ConfigSection>
        </div>
      )}

      {/* Delete Button */}
      {!isStartOrEnd && (
        <>
          <Separator />
          <Button
            variant="destructive"
            size="sm"
            className="w-full"
            onClick={onDelete}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete Node
          </Button>
        </>
      )}
    </div>
  );
}
