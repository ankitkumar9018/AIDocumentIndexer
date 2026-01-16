"use client";

import { useState } from "react";
import {
  Download,
  Monitor,
  Terminal,
  Globe,
  CheckCircle2,
  Copy,
  Apple,
  Laptop,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

export default function SyncClientsPage() {
  const [copiedCommand, setCopiedCommand] = useState<string | null>(null);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCommand(label);
    toast.success("Copied to clipboard");
    setTimeout(() => setCopiedCommand(null), 2000);
  };

  const serverUrl = typeof window !== "undefined" ? window.location.origin : "";
  const cliDownloadUrl = `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1"}/downloads/cli`;

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Download className="h-8 w-8" />
          Sync Clients
        </h1>
        <p className="text-muted-foreground mt-1">
          Download apps to automatically sync files from your computer to Mandala
        </p>
      </div>

      {/* Client Options */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Desktop App */}
        <Card className="relative">
          <Badge className="absolute top-4 right-4 bg-blue-500">Recommended</Badge>
          <CardHeader>
            <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mb-2">
              <Monitor className="h-6 w-6 text-white" />
            </div>
            <CardTitle>Desktop App</CardTitle>
            <CardDescription>
              Full-featured app with system tray, offline queue, and native notifications
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Available for:</p>
              <div className="flex gap-2">
                <Badge variant="outline" className="gap-1">
                  <Apple className="h-3 w-3" /> macOS
                </Badge>
                <Badge variant="outline" className="gap-1">
                  <Laptop className="h-3 w-3" /> Windows
                </Badge>
                <Badge variant="outline">Linux</Badge>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Runs in system tray
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Watch multiple folders
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Offline upload queue
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Native notifications
                </li>
              </ul>
            </div>
            <Button className="w-full" disabled>
              <Download className="h-4 w-4 mr-2" />
              Coming Soon
            </Button>
          </CardContent>
        </Card>

        {/* CLI Tool */}
        <Card>
          <CardHeader>
            <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center mb-2">
              <Terminal className="h-6 w-6 text-white" />
            </div>
            <CardTitle>CLI Tool</CardTitle>
            <CardDescription>
              Command-line tool for automation and power users
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Works on:</p>
              <div className="flex gap-2">
                <Badge variant="outline">macOS</Badge>
                <Badge variant="outline">Windows</Badge>
                <Badge variant="outline">Linux</Badge>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Run as daemon/service
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Scriptable & automatable
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Low resource usage
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Config file support
                </li>
              </ul>
            </div>
            <Button className="w-full" variant="outline" asChild>
              <a href={cliDownloadUrl} download="mandala-sync-cli.zip">
                <Download className="h-4 w-4 mr-2" />
                Download CLI Tool
              </a>
            </Button>
          </CardContent>
        </Card>

        {/* Browser Extension */}
        <Card>
          <CardHeader>
            <div className="h-12 w-12 rounded-lg bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center mb-2">
              <Globe className="h-6 w-6 text-white" />
            </div>
            <CardTitle>Browser Extension</CardTitle>
            <CardDescription>
              Auto-upload downloaded files directly from your browser
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Available for:</p>
              <div className="flex gap-2">
                <Badge variant="outline">Chrome</Badge>
                <Badge variant="outline">Firefox</Badge>
                <Badge variant="outline">Edge</Badge>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-medium">Features:</p>
              <ul className="text-sm text-muted-foreground space-y-1">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Watch downloads folder
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Right-click upload
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  File type filters
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-3 w-3 text-green-500" />
                  Upload notifications
                </li>
              </ul>
            </div>
            <Button className="w-full" disabled>
              <Download className="h-4 w-4 mr-2" />
              Coming Soon
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* CLI Installation Guide */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Terminal className="h-5 w-5" />
            CLI Quick Start
          </CardTitle>
          <CardDescription>
            Get started with the command-line tool in minutes
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="pip" className="w-full">
            <TabsList>
              <TabsTrigger value="pip">pip (Python)</TabsTrigger>
              <TabsTrigger value="brew">Homebrew</TabsTrigger>
              <TabsTrigger value="manual">Manual</TabsTrigger>
            </TabsList>

            <TabsContent value="pip" className="space-y-4">
              <div className="space-y-2">
                <p className="text-sm font-medium">1. Install the CLI tool:</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-muted p-3 rounded-md text-sm font-mono">
                    pip install mandala-sync
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard("pip install mandala-sync", "pip")}
                  >
                    {copiedCommand === "pip" ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">2. Login to your server:</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-muted p-3 rounded-md text-sm font-mono">
                    mandala-sync login --server {serverUrl || "https://your-server.com"}
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard(`mandala-sync login --server ${serverUrl || "https://your-server.com"}`, "login")}
                  >
                    {copiedCommand === "login" ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">3. Add a folder to watch:</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-muted p-3 rounded-md text-sm font-mono">
                    mandala-sync watch ~/Documents --collection &quot;My Docs&quot;
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard('mandala-sync watch ~/Documents --collection "My Docs"', "watch")}
                  >
                    {copiedCommand === "watch" ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm font-medium">4. Start syncing:</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-muted p-3 rounded-md text-sm font-mono">
                    mandala-sync start --scan
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard("mandala-sync start --scan", "start")}
                  >
                    {copiedCommand === "start" ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="brew" className="space-y-4">
              <div className="space-y-2">
                <p className="text-sm font-medium">Install via Homebrew (macOS/Linux):</p>
                <div className="flex items-center gap-2">
                  <code className="flex-1 bg-muted p-3 rounded-md text-sm font-mono">
                    brew install mandala-sync
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => copyToClipboard("brew install mandala-sync", "brew")}
                  >
                    {copiedCommand === "brew" ? (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    ) : (
                      <Copy className="h-4 w-4" />
                    )}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Coming soon - package not yet available
                </p>
              </div>
            </TabsContent>

            <TabsContent value="manual" className="space-y-4">
              <div className="space-y-2">
                <p className="text-sm font-medium">Download and install manually:</p>
                <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
                  <li>Download the CLI tool zip file</li>
                  <li>Extract the archive</li>
                  <li>Install with: <code className="bg-muted px-1 rounded">pip install .</code></li>
                </ol>
                <Button variant="outline" asChild>
                  <a href={cliDownloadUrl} download="mandala-sync-cli.zip">
                    <Download className="h-4 w-4 mr-2" />
                    Download CLI Tool
                  </a>
                </Button>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* All Commands Reference */}
      <Card>
        <CardHeader>
          <CardTitle>CLI Commands Reference</CardTitle>
          <CardDescription>
            Full list of available commands
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-3">
              <h4 className="font-medium">Authentication</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync login</code>
                  <span className="text-xs">Login to server</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync logout</code>
                  <span className="text-xs">Clear credentials</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync whoami</code>
                  <span className="text-xs">Show current user</span>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-medium">Directories</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync watch &lt;path&gt;</code>
                  <span className="text-xs">Add directory</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync unwatch &lt;path&gt;</code>
                  <span className="text-xs">Remove directory</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync list</code>
                  <span className="text-xs">List directories</span>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-medium">Syncing</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync start</code>
                  <span className="text-xs">Start watching</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync start --scan</code>
                  <span className="text-xs">Start + scan existing</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync scan &lt;path&gt;</code>
                  <span className="text-xs">Scan & upload folder</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync status</code>
                  <span className="text-xs">Show status</span>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-medium">Configuration</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync config get</code>
                  <span className="text-xs">Show config</span>
                </div>
                <div className="flex justify-between">
                  <code className="text-muted-foreground">mandala-sync config set &lt;key&gt; &lt;val&gt;</code>
                  <span className="text-xs">Set config value</span>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
