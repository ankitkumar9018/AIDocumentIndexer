# AIDocumentIndexer Documentation

Welcome to the AIDocumentIndexer documentation. This guide covers everything you need to know to use and develop with AIDocumentIndexer.

## Quick Navigation

### Tutorials (Step-by-Step Guides)
- [Quick Start](tutorials/01-quick-start.md) - Get up and running in 5 minutes
- [Bulk Processing](tutorials/06-bulk-processing.md) - Process 100K+ documents
- [Ray Scaling](tutorials/08-ray-scaling.md) - Horizontal scaling with Ray
- [Visual Documents](tutorials/10-visual-documents.md) - Process charts, tables, and images
- [All Tutorials](tutorials/README.md) - Complete tutorial list

### User Guide
- [Getting Started](user-guide/getting-started.md) - First steps with AIDocumentIndexer
- [Uploading Documents](user-guide/uploading-documents.md) - How to upload and manage documents
- [Querying Documents](user-guide/querying-documents.md) - Search and ask questions
- [Audio Overviews](user-guide/audio-overviews.md) - Generate audio summaries
- [Knowledge Graph](user-guide/knowledge-graph.md) - Explore document relationships
- [AI Agents](user-guide/ai-agents.md) - Create custom chatbots
- [Troubleshooting](user-guide/troubleshooting.md) - Common issues and solutions

### Developer Guide
- [Architecture](developer-guide/architecture.md) - System architecture overview
- [API Reference](developer-guide/api-reference.md) - REST API documentation
- [Contributing](developer-guide/contributing.md) - How to contribute
- [Testing](developer-guide/testing.md) - Testing guidelines
- [Deployment](developer-guide/deployment.md) - Deployment options

### Architecture Decision Records
- [ADR-001: Celery Task Queue](adrs/001-celery-task-queue.md)
- [ADR-002: ColBERT Retrieval](adrs/002-colbert-retrieval.md)
- [ADR-003: RLM Integration](adrs/003-rlm-integration.md)
- [ADR-004: Ray Distributed Computing](adrs/004-ray-distributed.md)
- [ADR-005: VLM Integration](adrs/005-vlm-integration.md)

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

### Answer Quality
- **Recursive LM**: 10M+ token context with O(log N) complexity
- **Self-Refine**: 20%+ quality improvement
- **Chain-of-Verification**: Hallucination reduction
- **Tree of Thoughts**: Complex reasoning
- **SELF-RAG**: Self-correcting retrieval

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

- **GitHub Issues**: [Report bugs](https://github.com/anthropics/claude-code/issues)
- **Documentation**: You're here!
- **API Reference**: See `/docs/developer-guide/api-reference.md`

## License

MIT License - See [LICENSE](../LICENSE) for details.
