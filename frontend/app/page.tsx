import Link from "next/link";
import { ArrowRight, FileText, MessageSquare, Upload, Shield, Sparkles, Globe } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted">
      {/* Header */}
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-6 w-6 text-primary" />
            <span className="text-xl font-bold">AIDocumentIndexer</span>
          </div>
          <nav className="flex items-center gap-4">
            <Link
              href="/login"
              className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Login
            </Link>
            <Link
              href="/login"
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90 transition-colors"
            >
              Get Started
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container py-24 md:py-32">
        <div className="mx-auto max-w-3xl text-center">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl">
            Transform Your Document Archive Into
            <span className="text-primary"> Intelligent Knowledge</span>
          </h1>
          <p className="mt-6 text-lg text-muted-foreground">
            Turn 25+ years of presentations, reports, and documents into a searchable AI assistant.
            Ask questions, generate new content, and discover insights from your organization&apos;s collective knowledge.
          </p>
          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link
              href="/login"
              className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-base font-medium text-primary-foreground shadow-lg hover:bg-primary/90 transition-all hover:scale-105"
            >
              Start Exploring
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
            <Link
              href="#features"
              className="inline-flex items-center justify-center rounded-md border border-input bg-background px-8 py-3 text-base font-medium shadow-sm hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="container py-24">
        <h2 className="text-3xl font-bold text-center mb-12">
          Everything You Need to Unlock Your Knowledge
        </h2>
        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          {/* Feature 1 */}
          <div className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <MessageSquare className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-xl font-semibold">AI-Powered Chat</h3>
            <p className="text-muted-foreground">
              Ask questions in natural language and get accurate answers with source citations.
              Know exactly which documents informed each response.
            </p>
          </div>

          {/* Feature 2 */}
          <div className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Upload className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-xl font-semibold">Universal File Support</h3>
            <p className="text-muted-foreground">
              Upload PDFs, PowerPoints, Word documents, spreadsheets, images, and more.
              Our AI understands them all, including scanned documents.
            </p>
          </div>

          {/* Feature 3 */}
          <div className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Sparkles className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-xl font-semibold">Content Generation</h3>
            <p className="text-muted-foreground">
              Create new presentations, reports, and documents inspired by your existing work.
              Human-in-the-loop workflow ensures quality output.
            </p>
          </div>

          {/* Feature 4 */}
          <div className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Shield className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-xl font-semibold">Enterprise Security</h3>
            <p className="text-muted-foreground">
              Role-based access control ensures sensitive documents stay protected.
              CEOs see everything, interns only access authorized files.
            </p>
          </div>

          {/* Feature 5 */}
          <div className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <Globe className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-xl font-semibold">Web Integration</h3>
            <p className="text-muted-foreground">
              Scrape and integrate content from websites. Combine internal knowledge
              with external research for comprehensive insights.
            </p>
          </div>

          {/* Feature 6 */}
          <div className="rounded-xl border bg-card p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
              <FileText className="h-6 w-6 text-primary" />
            </div>
            <h3 className="mb-2 text-xl font-semibold">Multi-Language</h3>
            <p className="text-muted-foreground">
              Search and interact in English, German, and more. Our AI understands
              and processes documents in over 100 languages.
            </p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="border-t bg-muted/50 py-24">
        <div className="container text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Transform Your Knowledge Base?
          </h2>
          <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join organizations that have turned decades of documents into instant, searchable knowledge.
            Start discovering insights you never knew existed.
          </p>
          <Link
            href="/login"
            className="inline-flex items-center justify-center rounded-md bg-primary px-8 py-3 text-base font-medium text-primary-foreground shadow-lg hover:bg-primary/90 transition-all hover:scale-105"
          >
            Get Started Now
            <ArrowRight className="ml-2 h-5 w-5" />
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8">
        <div className="container flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">
              AIDocumentIndexer
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            Built with care for teams who value their knowledge
          </p>
        </div>
      </footer>
    </div>
  );
}
