"use client";

import * as React from "react";
import { Download, FileJson, FileText, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useExportChatSession } from "@/lib/api/hooks";

interface ExportChatButtonProps {
  sessionId: string;
  disabled?: boolean;
}

export function ExportChatButton({ sessionId, disabled }: ExportChatButtonProps) {
  const [isExporting, setIsExporting] = React.useState(false);

  const exportSession = async (format: "json" | "markdown" | "txt") => {
    setIsExporting(true);
    try {
      const response = await fetch(
        `/api/v1/chat/sessions/${sessionId}/export?format=${format}`
      );

      if (!response.ok) {
        throw new Error("Export failed");
      }

      const data = await response.json();

      // Create blob and download
      const mimeTypes = {
        json: "application/json",
        markdown: "text/markdown",
        txt: "text/plain",
      };

      const extensions = {
        json: "json",
        markdown: "md",
        txt: "txt",
      };

      const blob = new Blob([data.content], { type: mimeTypes[format] });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chat-export-${sessionId.slice(0, 8)}.${extensions[format]}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed:", error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" disabled={disabled || isExporting}>
          {isExporting ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <Download className="h-5 w-5" />
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => exportSession("json")}>
          <FileJson className="mr-2 h-4 w-4" />
          Export as JSON
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => exportSession("markdown")}>
          <FileText className="mr-2 h-4 w-4" />
          Export as Markdown
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => exportSession("txt")}>
          <FileText className="mr-2 h-4 w-4" />
          Export as Text
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
