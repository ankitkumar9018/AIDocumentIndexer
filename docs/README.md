# AIDocumentIndexer Documentation

Welcome to the AIDocumentIndexer documentation. This guide covers everything you need to know to use and develop with AIDocumentIndexer.

## Quick Navigation

### Tutorials (Step-by-Step Guides)
- [Quick Start](tutorials/01-quick-start.md) - Get up and running in 5 minutes
- [Bulk Processing](tutorials/06-bulk-processing.md) - Process 100K+ documents
- [Ray Scaling](tutorials/10-ray-scaling.md) - Horizontal scaling with Ray
- [Visual Documents](tutorials/12-visual-documents.md) - Process charts, tables, and images
- [All Tutorials](tutorials/README.md) - Complete tutorial list

### User Guide
- [Getting Started](user-guide/getting-started.md) - First steps with AIDocumentIndexer
- [Features](FEATURES.md) - Comprehensive feature documentation
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions

### Developer Guide
- [Technical Architecture](TECHNICAL_ARCHITECTURE.md) - System architecture overview
- [API Reference](API.md) - REST API documentation
- [Developer Onboarding](DEVELOPER_ONBOARDING.md) - Getting started as a developer
- [Code Reference](CODE_REFERENCE.md) - Codebase navigation guide
- [Configuration](CONFIGURATION.md) - Configuration reference
- [Installation](INSTALLATION.md) - Installation instructions
- [Deployment](DEPLOYMENT.md) - Deployment options
- [Security](SECURITY.md) - Security documentation
- [Commands](COMMANDS.md) - CLI commands reference

### Specialized Guides
- [AI Agents](AGENTS.md) - Agent configuration and usage
- [OCR Configuration](OCR_CONFIGURATION.md) - OCR setup and tuning
- [Embeddings Guide](embeddings/EMBEDDING_MODELS.md) - Embedding provider reference
- [Knowledge Graph](knowledge-graph/KNOWLEDGE_GRAPH_COMPLETION.md) - KG implementation details

## Features Overview

### Document Processing
- **Bulk Upload**: Process 100,000+ files with parallel processing
- **Multi-Format Support**: PDF, Office docs, images, and more
- **Vision OCR**: 97.7% accuracy with Surya/Claude Vision
- **Smart Chunking**: 33x faster with Chonkie
- **VLM Processing**: Chart/table extraction with 40% improved accuracy

### Intelligent Search
- **Hybrid Retrieval**: ColBERT + dense vectors + BM25
- **Contextual Embeddings**: 67% error reduction
- **Knowledge Graph**: Entity extraction and relationships
- **RAPTOR**: Hierarchical document understanding
- **WARP Engine**: 3x faster multi-vector retrieval
- **Hallucination Detection**: Multi-signal grounding verification
- **Content Freshness**: Time-based document scoring

### Answer Quality
- **Recursive LM**: 10M+ token context with O(log N) complexity
- **Self-Refine**: 20%+ quality improvement
- **Chain-of-Verification**: Hallucination reduction
- **Tree of Thoughts**: Complex reasoning
- **SELF-RAG**: Self-correcting retrieval
- **DSPy Optimization**: Automated prompt compilation (Phase 93)
- **Inline Citations**: Numbered source references in responses

### Audio & Real-Time
- **Cartesia TTS**: 40ms time-to-first-audio
- **ElevenLabs Flash**: 75ms TTFB alternative
- **Streaming Pipeline**: Query while processing
- **Partial Queries**: Results in 5 seconds

### Enterprise Features
- **Multi-Tenant**: Organization isolation
- **RBAC**: Role-based access control
- **Audit Logging**: SOC2/GDPR compliance
- **AI Agents**: Custom chatbots with APIs

### UI & Developer Experience
- **Admin Settings**: 19-tab vertical navigation with category grouping
- **Dark Mode**: Full dark theme coverage
- **Canvas Panel**: Side-by-side artifact viewing and editing
- **Prompt Library**: Reusable template management
- **BYOK**: Bring Your Own API Key support

### Distributed Computing
- **Celery + Redis**: Priority task queues
- **Ray Integration**: Parallel ML workloads (10x throughput)
- **Actor Pools**: Stateful worker management
- **Auto-Fallback**: Graceful degradation

## Performance Targets

| Metric | Target |
|--------|--------|
| 100K file processing | 4-6 hours (with Ray) |
| User request latency | <200ms |
| Search latency (p95) | <200ms |
| Retrieval accuracy | 95%+ |
| TTS time-to-first-audio | 40ms |
| VLM visual accuracy | +40% |
| RLM context | 10M+ tokens |

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/anthropics/AIDocumentIndexer/issues)
- **Documentation**: You're here!
- **API Reference**: See [API.md](API.md)

## License

MIT License - See [LICENSE](../LICENSE) for details.
