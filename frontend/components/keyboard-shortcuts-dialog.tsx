"use client";

import { useState, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Keyboard } from "lucide-react";

interface ShortcutGroup {
  title: string;
  shortcuts: {
    keys: string[];
    description: string;
  }[];
}

const shortcutGroups: ShortcutGroup[] = [
  {
    title: "General",
    shortcuts: [
      { keys: ["?"], description: "Open keyboard shortcuts" },
      { keys: ["⌘", "/"], description: "Open keyboard shortcuts" },
      { keys: ["⌘", "K"], description: "Focus search input" },
      { keys: ["Esc"], description: "Close dialogs/panels" },
    ],
  },
  {
    title: "Chat",
    shortcuts: [
      { keys: ["⌘", "Enter"], description: "Send message" },
      { keys: ["⌘", "N"], description: "New chat" },
      { keys: ["⌘", "⇧", "E"], description: "Export chat" },
    ],
  },
  {
    title: "Documents",
    shortcuts: [
      { keys: ["⌘", "A"], description: "Select all documents" },
      { keys: ["⌘", "⇧", "D"], description: "Download selected" },
      { keys: ["Del"], description: "Delete selected" },
    ],
  },
  {
    title: "Navigation",
    shortcuts: [
      { keys: ["⌘", "1"], description: "Go to Dashboard" },
      { keys: ["⌘", "2"], description: "Go to Chat" },
      { keys: ["⌘", "3"], description: "Go to Documents" },
      { keys: ["⌘", "4"], description: "Go to Upload" },
    ],
  },
];

export function KeyboardShortcutsDialog() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Open with ? or Cmd+/
      if (e.key === "?" || ((e.metaKey || e.ctrlKey) && e.key === "/")) {
        e.preventDefault();
        setOpen(true);
      }
      // Close with Escape
      if (e.key === "Escape") {
        setOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Keyboard className="h-5 w-5" />
            Keyboard Shortcuts
          </DialogTitle>
          <DialogDescription>
            Use these shortcuts to navigate and interact with the app more efficiently.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 md:grid-cols-2 mt-4">
          {shortcutGroups.map((group) => (
            <div key={group.title}>
              <h3 className="text-sm font-semibold text-muted-foreground mb-3">
                {group.title}
              </h3>
              <div className="space-y-2">
                {group.shortcuts.map((shortcut, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between py-1.5 px-2 rounded-md hover:bg-muted"
                  >
                    <span className="text-sm">{shortcut.description}</span>
                    <div className="flex items-center gap-1">
                      {shortcut.keys.map((key, keyIdx) => (
                        <Badge
                          key={keyIdx}
                          variant="outline"
                          className="h-6 min-w-6 px-1.5 font-mono text-xs"
                        >
                          {key}
                        </Badge>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-4 pt-4 border-t text-center">
          <p className="text-xs text-muted-foreground">
            Press <Badge variant="outline" className="h-5 px-1.5 font-mono text-xs mx-1">Esc</Badge> to close
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
