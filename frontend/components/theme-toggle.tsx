"use client";

import * as React from "react";
import { Moon, Sun, Monitor } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  // Avoid hydration mismatch
  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Button variant="ghost" size="icon" className="h-9 w-9">
        <Sun className="h-4 w-4" />
      </Button>
    );
  }

  const cycleTheme = () => {
    if (theme === "light") {
      setTheme("dark");
    } else if (theme === "dark") {
      setTheme("system");
    } else {
      setTheme("light");
    }
  };

  return (
    <Button
      variant="ghost"
      size="icon"
      className="h-9 w-9"
      onClick={cycleTheme}
      title={`Current: ${theme}. Click to change.`}
    >
      {theme === "light" && <Sun className="h-4 w-4" />}
      {theme === "dark" && <Moon className="h-4 w-4" />}
      {theme === "system" && <Monitor className="h-4 w-4" />}
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}

export function ThemeToggleDropdown() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <div className="flex items-center gap-1 p-1 rounded-lg bg-muted">
      <Button
        variant={theme === "light" ? "secondary" : "ghost"}
        size="sm"
        className="h-7 px-2"
        onClick={() => setTheme("light")}
      >
        <Sun className="h-3.5 w-3.5 mr-1" />
        Light
      </Button>
      <Button
        variant={theme === "dark" ? "secondary" : "ghost"}
        size="sm"
        className="h-7 px-2"
        onClick={() => setTheme("dark")}
      >
        <Moon className="h-3.5 w-3.5 mr-1" />
        Dark
      </Button>
      <Button
        variant={theme === "system" ? "secondary" : "ghost"}
        size="sm"
        className="h-7 px-2"
        onClick={() => setTheme("system")}
      >
        <Monitor className="h-3.5 w-3.5 mr-1" />
        Auto
      </Button>
    </div>
  );
}
