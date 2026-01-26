"use client";

/**
 * AIDocumentIndexer - Help Center
 * ================================
 *
 * Phase 26: In-app help center component for documentation and guidance.
 *
 * Features:
 * - Searchable help articles
 * - Categorized content (Getting Started, Features, Troubleshooting, FAQ)
 * - Video tutorials integration
 * - Keyboard shortcuts reference
 * - Contact support
 * - Contextual help based on current page
 */

import * as React from "react";
import { useState, useMemo, useCallback } from "react";
import {
  Book,
  ChevronRight,
  ExternalLink,
  FileQuestion,
  HelpCircle,
  Keyboard,
  LifeBuoy,
  Lightbulb,
  MessageCircle,
  Play,
  Rocket,
  Search,
  Settings,
  Sparkles,
  Upload,
  Video,
  X,
  Zap,
  BookOpen,
  FileText,
  Headphones,
  BarChart3,
  Network,
  Shield,
  Users,
  Clock,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

// =============================================================================
// Types
// =============================================================================

export interface HelpArticle {
  id: string;
  title: string;
  description: string;
  content: string;
  category: HelpCategory;
  tags: string[];
  icon?: React.ReactNode;
  videoUrl?: string;
  relatedArticles?: string[];
}

export type HelpCategory =
  | "getting-started"
  | "features"
  | "troubleshooting"
  | "faq"
  | "shortcuts"
  | "api";

interface HelpCenterProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialCategory?: HelpCategory;
  initialArticleId?: string;
  currentPage?: string; // For contextual help
}

// =============================================================================
// Help Content Data
// =============================================================================

const HELP_ARTICLES: HelpArticle[] = [
  // Getting Started
  {
    id: "quick-start",
    title: "Quick Start Guide",
    description: "Get up and running in 5 minutes",
    content: `
## Welcome to AIDocumentIndexer!

Follow these steps to get started:

### 1. Upload Your Documents
- Click the **Upload** button or drag & drop files
- Supported formats: PDF, Word, PowerPoint, Excel, Text, Markdown
- Maximum file size: 100MB per file

### 2. Wait for Processing
- Documents are automatically extracted and indexed
- You can start asking questions even while processing continues

### 3. Ask Questions
- Type your question in the chat interface
- Get instant answers with source references
- Click on sources to view the original document

### 4. Explore More Features
- Generate audio summaries
- Create reports and presentations
- Visualize knowledge graphs

**Pro Tip:** Press \`?\` anytime to see keyboard shortcuts!
    `,
    category: "getting-started",
    tags: ["start", "upload", "documents", "chat", "beginner"],
    icon: <Rocket className="h-5 w-5" />,
    relatedArticles: ["upload-documents", "chat-basics", "audio-overview"],
  },
  {
    id: "upload-documents",
    title: "Uploading Documents",
    description: "Learn how to add documents to your knowledge base",
    content: `
## Uploading Documents

### Supported File Types
- **PDF** - All versions supported, including scanned documents (OCR)
- **Word** - .doc and .docx files
- **PowerPoint** - .ppt and .pptx presentations
- **Excel** - .xls and .xlsx spreadsheets
- **Text** - .txt, .md, .csv files

### Upload Methods

#### 1. Drag & Drop
Simply drag files from your computer and drop them on the upload area.

#### 2. File Browser
Click the upload area to open your system's file browser.

#### 3. Google Drive
Connect your Google Drive to import documents directly.

#### 4. URL Import
Paste a URL to import web pages as documents.

### Bulk Upload
Upload up to 1,000 files at once. Progress is tracked in real-time.

### Processing Time
- Text files: Instant
- PDFs (text): 5-10 seconds
- PDFs (scanned): 30-60 seconds (OCR)
- Large files: May take several minutes

**Note:** You can start asking questions immediately after your first document is processed!
    `,
    category: "getting-started",
    tags: ["upload", "documents", "pdf", "import", "google drive"],
    icon: <Upload className="h-5 w-5" />,
  },
  {
    id: "chat-basics",
    title: "Chat Interface Basics",
    description: "Learn how to interact with your documents",
    content: `
## Chat Interface

### Asking Questions
Type natural language questions like:
- "What are the main findings in the quarterly report?"
- "Summarize the key points from the meeting notes"
- "What does the contract say about termination clauses?"

### Source References
- Answers include clickable source references
- Click a source to view the exact passage in the document
- Sources are ranked by relevance

### Chat Features
- **Voice Input**: Click the microphone icon to speak your question
- **Text-to-Speech**: Listen to answers with the speaker icon
- **Export**: Download conversations as PDF or Markdown
- **Templates**: Use pre-built prompt templates for common tasks

### Advanced Queries
- Use **@mentions** to reference specific documents
- Filter by collection or document type
- Use quotation marks for exact phrase matching

### Keyboard Shortcuts
- \`Cmd/Ctrl + Enter\`: Send message
- \`Cmd/Ctrl + N\`: New chat
- \`Cmd/Ctrl + K\`: Focus search
    `,
    category: "getting-started",
    tags: ["chat", "questions", "answers", "voice", "sources"],
    icon: <MessageCircle className="h-5 w-5" />,
  },

  // Features
  {
    id: "audio-overview",
    title: "Audio Document Overview",
    description: "Generate podcast-style summaries of your documents",
    content: `
## Audio Overview Feature

Transform your documents into engaging audio summaries!

### How It Works
1. Select documents or a collection
2. Click **Generate Audio Overview**
3. Wait for AI to create a summary
4. Listen to a podcast-style audio overview

### Audio Formats
- **Brief** (2-3 minutes): Quick highlights
- **Standard** (5-10 minutes): Comprehensive overview
- **Deep Dive** (15-20 minutes): Detailed analysis

### Voice Options
- Multiple voice styles available
- Adjust speed and pitch
- Download as MP3 for offline listening

### Use Cases
- Review meeting notes on your commute
- Get summaries of long reports
- Create accessible versions of documents
- Study materials on the go

**Tip:** Audio summaries work best with text-heavy documents like reports and articles.
    `,
    category: "features",
    tags: ["audio", "podcast", "voice", "summary", "tts"],
    icon: <Headphones className="h-5 w-5" />,
  },
  {
    id: "knowledge-graph",
    title: "Knowledge Graph Visualization",
    description: "Explore connections between concepts in your documents",
    content: `
## Knowledge Graph

Visualize relationships between entities in your document collection.

### What It Shows
- **Entities**: People, organizations, concepts, dates, locations
- **Relationships**: How entities are connected
- **Clusters**: Groups of related concepts

### Interacting with the Graph
- **Zoom**: Scroll or pinch to zoom in/out
- **Pan**: Click and drag to move around
- **Select**: Click a node to see details
- **Filter**: Use the sidebar to filter by entity type

### Graph Analysis
- Find hidden connections between documents
- Discover key influencers or concepts
- Trace information flow
- Identify knowledge gaps

### Export Options
- Export as PNG/SVG image
- Download graph data as JSON
- Share interactive view with link

**Note:** Knowledge graph extraction happens automatically after document upload and may take a few minutes for large collections.
    `,
    category: "features",
    tags: ["knowledge graph", "visualization", "entities", "relationships"],
    icon: <Network className="h-5 w-5" />,
  },
  {
    id: "document-generation",
    title: "Document Generation",
    description: "Create reports, presentations, and more from your knowledge base",
    content: `
## Document Generation

Generate new documents based on your indexed content.

### Document Types
- **Reports**: Comprehensive written reports
- **Presentations**: PowerPoint slides with speaker notes
- **Summaries**: Executive summaries and briefs
- **Q&A Documents**: FAQ-style documents

### Templates
- Pre-built templates for common document types
- Customize sections and formatting
- Save your own templates

### Generation Options
- **Length**: Brief, Standard, Detailed
- **Tone**: Formal, Casual, Technical
- **Audience**: Executive, Technical, General

### Review & Edit
- Preview generated content before export
- Edit sections individually
- Regenerate specific sections
- Add manual edits

### Export Formats
- PDF
- Word (.docx)
- PowerPoint (.pptx)
- Markdown
- HTML
    `,
    category: "features",
    tags: ["generation", "reports", "presentations", "export"],
    icon: <FileText className="h-5 w-5" />,
  },
  {
    id: "collections",
    title: "Managing Collections",
    description: "Organize documents into collections for focused analysis",
    content: `
## Collections

Organize your documents into logical groups for better search and analysis.

### Creating Collections
1. Click **New Collection** in the sidebar
2. Name your collection
3. Add description (optional)
4. Set permissions (private/shared)

### Adding Documents
- Drag documents into a collection
- Use the move/copy menu
- Select multiple documents with Cmd/Ctrl + Click

### Collection Features
- **Focused Chat**: Ask questions within a specific collection
- **Bulk Operations**: Process all documents at once
- **Sharing**: Share collections with team members
- **Statistics**: View collection analytics

### Smart Collections
Create automatic collections based on rules:
- Document type
- Upload date
- Content keywords
- Author/source

### Permissions
- **Private**: Only you can access
- **Shared**: Team members can view
- **Editable**: Team members can modify
    `,
    category: "features",
    tags: ["collections", "organize", "folders", "sharing"],
    icon: <BookOpen className="h-5 w-5" />,
  },

  // Troubleshooting
  {
    id: "upload-fails",
    title: "Document Upload Fails",
    description: "Troubleshoot common upload issues",
    content: `
## Upload Troubleshooting

### Common Issues

#### File Too Large
- Maximum file size: 100MB
- **Solution**: Split large documents or compress them

#### Unsupported Format
- Check the supported formats list
- **Solution**: Convert to PDF or supported format

#### Corrupted File
- File may be damaged or encrypted
- **Solution**: Try re-downloading or re-exporting the file

#### Password Protected
- We can't process encrypted documents
- **Solution**: Remove password protection before upload

#### Network Error
- Connection may have been interrupted
- **Solution**: Check your connection and try again

### Still Having Issues?

1. Clear your browser cache
2. Try a different browser
3. Check if the file opens locally
4. Contact support with error details

**Note:** Processing times vary based on document complexity and server load.
    `,
    category: "troubleshooting",
    tags: ["upload", "error", "failed", "issues", "troubleshoot"],
    icon: <AlertCircle className="h-5 w-5" />,
  },
  {
    id: "slow-responses",
    title: "Slow Response Times",
    description: "Improve chat and search performance",
    content: `
## Improving Performance

### Factors Affecting Speed
- Number of documents in search scope
- Query complexity
- Server load
- Network conditions

### Quick Fixes

#### 1. Narrow Your Search
- Use collections to limit scope
- Filter by document type or date
- Use specific keywords

#### 2. Simplify Questions
- Break complex questions into parts
- Be specific about what you need
- Avoid very broad queries

#### 3. Check Your Connection
- Test your internet speed
- Try a wired connection
- Disable VPN if possible

### Expected Response Times
- Simple queries: 1-3 seconds
- Complex analysis: 5-10 seconds
- Document generation: 30-60 seconds
- Audio generation: 1-5 minutes

### During Peak Hours
- Response times may be longer
- Consider scheduling batch operations
- Premium users get priority processing
    `,
    category: "troubleshooting",
    tags: ["slow", "performance", "speed", "latency"],
    icon: <Clock className="h-5 w-5" />,
  },
  {
    id: "inaccurate-answers",
    title: "Getting Better Answers",
    description: "Tips for more accurate and relevant responses",
    content: `
## Improving Answer Quality

### Writing Better Questions

#### Be Specific
- ❌ "Tell me about the project"
- ✅ "What are the Q3 milestones for Project Alpha?"

#### Provide Context
- ❌ "What does it say about pricing?"
- ✅ "What does the 2024 pricing document say about enterprise tier pricing?"

#### Use @Mentions
Reference specific documents:
- "@contract.pdf what are the termination clauses?"

### Understanding Sources
- Check the confidence score
- Verify information in original document
- Look for multiple supporting sources

### When Answers Are Wrong
1. Rephrase your question
2. Check if relevant documents are uploaded
3. Review document quality (OCR accuracy)
4. Report the issue for improvement

### Document Quality Matters
- Clear, well-formatted documents work best
- Scanned documents may have OCR errors
- Tables and complex layouts may lose formatting
    `,
    category: "troubleshooting",
    tags: ["answers", "accuracy", "quality", "questions"],
    icon: <Lightbulb className="h-5 w-5" />,
  },

  // FAQ
  {
    id: "faq-privacy",
    title: "How is my data protected?",
    description: "Learn about our security and privacy measures",
    content: `
## Data Privacy & Security

### Data Encryption
- **At Rest**: AES-256 encryption
- **In Transit**: TLS 1.3
- **Processing**: Secure enclaves

### Data Retention
- Documents stored until you delete them
- Chat history retained for 90 days by default
- You can request data deletion anytime

### Access Control
- Role-based permissions
- Audit logs for all actions
- Two-factor authentication available

### Compliance
- SOC 2 Type II certified
- GDPR compliant
- HIPAA ready (enterprise plan)

### Third-Party Access
- We don't sell your data
- AI models don't train on your data
- Minimal data shared with cloud providers

### Data Location
- Primary: AWS US regions
- EU data available on enterprise plans
- Contact us for specific requirements
    `,
    category: "faq",
    tags: ["privacy", "security", "data", "encryption", "gdpr"],
    icon: <Shield className="h-5 w-5" />,
  },
  {
    id: "faq-limits",
    title: "What are the usage limits?",
    description: "Understand plan limits and quotas",
    content: `
## Usage Limits

### Free Plan
- 10 documents
- 50 queries per day
- Basic features only
- Community support

### Pro Plan
- 500 documents
- Unlimited queries
- All features
- Email support

### Enterprise Plan
- Unlimited documents
- Unlimited queries
- Custom integrations
- Dedicated support
- SSO & advanced security

### Query Limits
- Rate limit: 60 queries/minute
- Max context: 100,000 tokens
- Max response: 8,000 tokens

### File Limits
- Max file size: 100MB
- Max pages per PDF: 500
- Bulk upload: 1,000 files

### Need More?
Contact sales for custom limits.
    `,
    category: "faq",
    tags: ["limits", "quota", "plan", "pricing"],
    icon: <BarChart3 className="h-5 w-5" />,
  },
  {
    id: "faq-team",
    title: "How do I add team members?",
    description: "Invite and manage team collaboration",
    content: `
## Team Collaboration

### Inviting Members
1. Go to **Settings** → **Team**
2. Click **Invite Member**
3. Enter email address
4. Select role (Viewer, Editor, Admin)
5. Send invitation

### Roles & Permissions

#### Viewer
- View documents and collections
- Ask questions
- Cannot upload or modify

#### Editor
- Everything Viewers can do
- Upload documents
- Create collections
- Cannot manage team

#### Admin
- Everything Editors can do
- Manage team members
- Access billing
- Configure settings

### Shared Collections
- Create shared collections for team access
- Set collection-level permissions
- Track who made changes with audit logs

### Organization Features (Enterprise)
- Multiple teams
- Custom roles
- SSO integration
- Centralized billing
    `,
    category: "faq",
    tags: ["team", "collaboration", "invite", "sharing", "roles"],
    icon: <Users className="h-5 w-5" />,
  },
];

// Keyboard shortcuts data
const KEYBOARD_SHORTCUTS = [
  {
    category: "General",
    shortcuts: [
      { keys: ["?"], description: "Open keyboard shortcuts" },
      { keys: ["⌘", "/"], description: "Open keyboard shortcuts" },
      { keys: ["⌘", "K"], description: "Open command palette" },
      { keys: ["Esc"], description: "Close dialogs/panels" },
      { keys: ["F1"], description: "Open help center" },
    ],
  },
  {
    category: "Chat",
    shortcuts: [
      { keys: ["⌘", "Enter"], description: "Send message" },
      { keys: ["⌘", "N"], description: "New chat" },
      { keys: ["⌘", "⇧", "E"], description: "Export chat" },
      { keys: ["⌘", "⇧", "V"], description: "Toggle voice mode" },
    ],
  },
  {
    category: "Documents",
    shortcuts: [
      { keys: ["⌘", "U"], description: "Upload documents" },
      { keys: ["⌘", "A"], description: "Select all documents" },
      { keys: ["⌘", "⇧", "D"], description: "Download selected" },
      { keys: ["Del"], description: "Delete selected" },
    ],
  },
  {
    category: "Navigation",
    shortcuts: [
      { keys: ["⌘", "1"], description: "Go to Dashboard" },
      { keys: ["⌘", "2"], description: "Go to Chat" },
      { keys: ["⌘", "3"], description: "Go to Documents" },
      { keys: ["⌘", "4"], description: "Go to Upload" },
      { keys: ["⌘", "5"], description: "Go to Settings" },
    ],
  },
];

// Category configuration
const CATEGORY_CONFIG: Record<HelpCategory, { label: string; icon: React.ReactNode }> = {
  "getting-started": { label: "Getting Started", icon: <Rocket className="h-4 w-4" /> },
  features: { label: "Features", icon: <Sparkles className="h-4 w-4" /> },
  troubleshooting: { label: "Troubleshooting", icon: <LifeBuoy className="h-4 w-4" /> },
  faq: { label: "FAQ", icon: <FileQuestion className="h-4 w-4" /> },
  shortcuts: { label: "Shortcuts", icon: <Keyboard className="h-4 w-4" /> },
  api: { label: "API Reference", icon: <Settings className="h-4 w-4" /> },
};

// =============================================================================
// Help Center Component
// =============================================================================

export function HelpCenter({
  open,
  onOpenChange,
  initialCategory = "getting-started",
  initialArticleId,
  currentPage,
}: HelpCenterProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<HelpCategory>(initialCategory);
  const [selectedArticle, setSelectedArticle] = useState<HelpArticle | null>(
    initialArticleId ? HELP_ARTICLES.find((a) => a.id === initialArticleId) || null : null
  );

  // Filter articles based on search
  const filteredArticles = useMemo(() => {
    if (!searchQuery.trim()) {
      return HELP_ARTICLES.filter((a) => a.category === selectedCategory);
    }

    const query = searchQuery.toLowerCase();
    return HELP_ARTICLES.filter(
      (article) =>
        article.title.toLowerCase().includes(query) ||
        article.description.toLowerCase().includes(query) ||
        article.tags.some((tag) => tag.toLowerCase().includes(query))
    );
  }, [searchQuery, selectedCategory]);

  // Get contextual suggestions based on current page
  const contextualSuggestions = useMemo(() => {
    if (!currentPage) return [];

    const pageArticles: Record<string, string[]> = {
      "/chat": ["chat-basics", "inaccurate-answers", "audio-overview"],
      "/upload": ["upload-documents", "upload-fails", "collections"],
      "/documents": ["collections", "upload-documents"],
      "/dashboard": ["quick-start", "knowledge-graph"],
      "/settings": ["faq-team", "faq-privacy"],
    };

    const articleIds = pageArticles[currentPage] || ["quick-start"];
    return HELP_ARTICLES.filter((a) => articleIds.includes(a.id));
  }, [currentPage]);

  const handleArticleClick = useCallback((article: HelpArticle) => {
    setSelectedArticle(article);
  }, []);

  const handleBack = useCallback(() => {
    setSelectedArticle(null);
  }, []);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[80vh] flex flex-col p-0 gap-0">
        <DialogHeader className="px-6 py-4 border-b shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <HelpCircle className="h-5 w-5 text-primary" />
              <DialogTitle>Help Center</DialogTitle>
            </div>
          </div>
          <DialogDescription>
            Search for help articles or browse by category
          </DialogDescription>

          {/* Search */}
          <div className="relative mt-4">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search help articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        </DialogHeader>

        <div className="flex-1 flex overflow-hidden">
          {selectedArticle ? (
            // Article View
            <div className="flex-1 flex flex-col">
              <div className="px-6 py-3 border-b flex items-center gap-2">
                <Button variant="ghost" size="sm" onClick={handleBack}>
                  <ChevronRight className="h-4 w-4 rotate-180 mr-1" />
                  Back
                </Button>
                <span className="text-muted-foreground">|</span>
                <Badge variant="secondary">
                  {CATEGORY_CONFIG[selectedArticle.category]?.label}
                </Badge>
              </div>
              <ScrollArea className="flex-1 px-6 py-4">
                <ArticleContent article={selectedArticle} />
              </ScrollArea>
            </div>
          ) : (
            // Browse View
            <>
              {/* Sidebar */}
              <div className="w-56 border-r p-4 shrink-0">
                <nav className="space-y-1">
                  {(Object.keys(CATEGORY_CONFIG) as HelpCategory[]).map((category) => (
                    <button
                      key={category}
                      onClick={() => {
                        setSelectedCategory(category);
                        setSearchQuery("");
                      }}
                      className={cn(
                        "w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors",
                        selectedCategory === category
                          ? "bg-primary/10 text-primary font-medium"
                          : "hover:bg-muted text-muted-foreground"
                      )}
                    >
                      {CATEGORY_CONFIG[category].icon}
                      {CATEGORY_CONFIG[category].label}
                    </button>
                  ))}
                </nav>

                {/* Quick Links */}
                <div className="mt-6 pt-6 border-t">
                  <p className="text-xs font-medium text-muted-foreground mb-2">
                    Quick Links
                  </p>
                  <div className="space-y-1">
                    <a
                      href="#"
                      className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground rounded-md hover:bg-muted"
                    >
                      <Video className="h-4 w-4" />
                      Video Tutorials
                      <ExternalLink className="h-3 w-3 ml-auto" />
                    </a>
                    <a
                      href="#"
                      className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground rounded-md hover:bg-muted"
                    >
                      <Book className="h-4 w-4" />
                      API Docs
                      <ExternalLink className="h-3 w-3 ml-auto" />
                    </a>
                    <a
                      href="#"
                      className="flex items-center gap-2 px-3 py-2 text-sm text-muted-foreground hover:text-foreground rounded-md hover:bg-muted"
                    >
                      <MessageCircle className="h-4 w-4" />
                      Contact Support
                    </a>
                  </div>
                </div>
              </div>

              {/* Content */}
              <ScrollArea className="flex-1 p-6">
                {selectedCategory === "shortcuts" ? (
                  <ShortcutsView />
                ) : (
                  <>
                    {/* Contextual Suggestions */}
                    {contextualSuggestions.length > 0 && !searchQuery && (
                      <div className="mb-6">
                        <h3 className="text-sm font-medium text-muted-foreground mb-3">
                          Suggested for this page
                        </h3>
                        <div className="grid gap-2">
                          {contextualSuggestions.map((article) => (
                            <ArticleCard
                              key={article.id}
                              article={article}
                              onClick={() => handleArticleClick(article)}
                              compact
                            />
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Category Articles */}
                    <div>
                      {searchQuery && (
                        <p className="text-sm text-muted-foreground mb-4">
                          {filteredArticles.length} results for "{searchQuery}"
                        </p>
                      )}
                      <div className="grid gap-3">
                        {filteredArticles.map((article) => (
                          <ArticleCard
                            key={article.id}
                            article={article}
                            onClick={() => handleArticleClick(article)}
                          />
                        ))}
                        {filteredArticles.length === 0 && (
                          <div className="text-center py-8 text-muted-foreground">
                            <FileQuestion className="h-8 w-8 mx-auto mb-2 opacity-50" />
                            <p>No articles found</p>
                            <p className="text-sm">Try a different search term</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </ScrollArea>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

// =============================================================================
// Sub-Components
// =============================================================================

function ArticleCard({
  article,
  onClick,
  compact,
}: {
  article: HelpArticle;
  onClick: () => void;
  compact?: boolean;
}) {
  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:border-primary/50 hover:shadow-sm",
        compact && "border-0 shadow-none bg-muted/50"
      )}
      onClick={onClick}
    >
      <CardContent className={cn("flex items-start gap-3", compact ? "p-3" : "p-4")}>
        <div
          className={cn(
            "shrink-0 rounded-lg flex items-center justify-center text-primary",
            compact ? "h-8 w-8 bg-primary/10" : "h-10 w-10 bg-primary/10"
          )}
        >
          {article.icon || <FileText className={compact ? "h-4 w-4" : "h-5 w-5"} />}
        </div>
        <div className="flex-1 min-w-0">
          <h4 className={cn("font-medium", compact && "text-sm")}>{article.title}</h4>
          <p className={cn("text-muted-foreground", compact ? "text-xs" : "text-sm")}>
            {article.description}
          </p>
          {!compact && article.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {article.tags.slice(0, 3).map((tag) => (
                <Badge key={tag} variant="outline" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          )}
        </div>
        <ChevronRight className="h-4 w-4 text-muted-foreground shrink-0 mt-1" />
      </CardContent>
    </Card>
  );
}

function ArticleContent({ article }: { article: HelpArticle }) {
  return (
    <div className="max-w-2xl">
      <div className="flex items-center gap-3 mb-4">
        <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
          {article.icon || <FileText className="h-6 w-6" />}
        </div>
        <div>
          <h2 className="text-xl font-semibold">{article.title}</h2>
          <p className="text-sm text-muted-foreground">{article.description}</p>
        </div>
      </div>

      {article.videoUrl && (
        <Card className="mb-6 overflow-hidden">
          <div className="aspect-video bg-muted flex items-center justify-center">
            <Button variant="outline" size="lg" className="gap-2">
              <Play className="h-5 w-5" />
              Watch Video Tutorial
            </Button>
          </div>
        </Card>
      )}

      <div className="prose prose-sm dark:prose-invert max-w-none">
        {/* Render markdown content */}
        <div
          dangerouslySetInnerHTML={{
            __html: article.content
              .replace(/^### (.*)$/gm, '<h3 class="text-base font-semibold mt-6 mb-2">$1</h3>')
              .replace(/^## (.*)$/gm, '<h2 class="text-lg font-semibold mt-8 mb-3">$1</h2>')
              .replace(/^#### (.*)$/gm, '<h4 class="text-sm font-semibold mt-4 mb-2">$1</h4>')
              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              .replace(/`([^`]+)`/g, '<code class="px-1.5 py-0.5 bg-muted rounded text-sm">$1</code>')
              .replace(/^- (.*)$/gm, '<li class="ml-4">$1</li>')
              .replace(/(<li.*<\/li>\n?)+/g, '<ul class="list-disc space-y-1 my-2">$&</ul>')
              .replace(/^(\d+)\. (.*)$/gm, '<li class="ml-4"><span class="font-medium">$1.</span> $2</li>')
              .replace(/\n\n/g, '</p><p class="my-3">')
              .replace(/❌/g, '<span class="text-destructive">❌</span>')
              .replace(/✅/g, '<span class="text-green-500">✅</span>'),
          }}
        />
      </div>

      {article.relatedArticles && article.relatedArticles.length > 0 && (
        <div className="mt-8 pt-6 border-t">
          <h3 className="text-sm font-medium text-muted-foreground mb-3">
            Related Articles
          </h3>
          <div className="space-y-2">
            {article.relatedArticles.map((id) => {
              const related = HELP_ARTICLES.find((a) => a.id === id);
              if (!related) return null;
              return (
                <a
                  key={id}
                  href="#"
                  className="flex items-center gap-2 text-sm text-primary hover:underline"
                >
                  <ChevronRight className="h-3 w-3" />
                  {related.title}
                </a>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function ShortcutsView() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-2">Keyboard Shortcuts</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Use these shortcuts to navigate and interact more efficiently.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {KEYBOARD_SHORTCUTS.map((group) => (
          <div key={group.category}>
            <h4 className="text-sm font-medium text-muted-foreground mb-3">
              {group.category}
            </h4>
            <div className="space-y-2">
              {group.shortcuts.map((shortcut, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between py-2 px-3 rounded-md hover:bg-muted"
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

      <div className="mt-6 p-4 bg-muted rounded-lg">
        <div className="flex items-start gap-3">
          <Lightbulb className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium">Pro Tip</p>
            <p className="text-sm text-muted-foreground">
              Press <Badge variant="outline" className="mx-1 font-mono">?</Badge> anywhere
              in the app to quickly view keyboard shortcuts.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Help Trigger Button
// =============================================================================

export function HelpTrigger({
  className,
  variant = "ghost",
}: {
  className?: string;
  variant?: "ghost" | "outline" | "default";
}) {
  const [open, setOpen] = useState(false);

  // Open help center with F1
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "F1") {
        e.preventDefault();
        setOpen(true);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <>
      <Button
        variant={variant}
        size="icon"
        onClick={() => setOpen(true)}
        className={className}
        aria-label="Help"
      >
        <HelpCircle className="h-5 w-5" />
      </Button>
      <HelpCenter open={open} onOpenChange={setOpen} />
    </>
  );
}

export default HelpCenter;
