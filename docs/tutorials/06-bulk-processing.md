# Tutorial: Bulk Processing 100K+ Documents

Learn how to process large document collections efficiently using AIDocumentIndexer's distributed processing capabilities.

## Overview

AIDocumentIndexer can process 100,000+ documents in 8-12 hours (compared to 50+ hours with sequential processing). This is achieved through:

- **Celery Task Queue**: Priority-based task scheduling
- **Ray Distributed Computing**: Parallel ML workloads
- **Bulk Progress Tracker**: Real-time monitoring

## Prerequisites

- Docker and Docker Compose running
- Redis server running
- Celery workers started (or Ray cluster for ML workloads)

## Step 1: Start the Processing Infrastructure

```bash
# Start Redis and PostgreSQL
docker-compose up -d redis postgres

# Start Celery workers (4 concurrent workers)
celery -A backend.services.task_queue worker -l info -c 4

# Optional: Start Ray cluster for ML workloads
ray start --head --num-cpus=8
```

## Step 2: Upload Documents via API

### Using the Bulk Upload Endpoint

```python
import httpx
import asyncio
from pathlib import Path

async def bulk_upload(folder_path: str, collection: str = "my-collection"):
    """Upload all documents from a folder."""
    files = list(Path(folder_path).glob("**/*.*"))

    # Filter supported formats
    supported = {".pdf", ".docx", ".txt", ".md", ".html"}
    files = [f for f in files if f.suffix.lower() in supported]

    print(f"Found {len(files)} documents to upload")

    # Prepare multipart form data
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Upload in batches of 100
        batch_size = 100

        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]

            form_data = [
                ("files", (f.name, open(f, "rb")))
                for f in batch
            ]
            form_data.append(("collection", collection))

            response = await client.post(
                "http://localhost:8000/api/upload/bulk",
                files=form_data
            )

            result = response.json()
            batch_id = result["batch_id"]
            print(f"Batch {i//batch_size + 1}: {batch_id}")

            # Close file handles
            for _, (_, file) in form_data[:-1]:
                file.close()

# Run
asyncio.run(bulk_upload("/path/to/documents", "legal-docs"))
```

## Step 3: Monitor Progress

### Using the Progress Endpoint

```python
async def monitor_progress(batch_id: str):
    """Monitor batch processing progress."""
    async with httpx.AsyncClient() as client:
        while True:
            response = await client.get(
                f"http://localhost:8000/api/upload/batch/{batch_id}/progress"
            )
            progress = response.json()

            print(f"Progress: {progress['processed']}/{progress['total']}")
            print(f"  - Completed: {progress['completed']}")
            print(f"  - Failed: {progress['failed']}")
            print(f"  - Current stage: {progress.get('current_stage', 'N/A')}")

            if progress["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(5)

    return progress
```

### Using WebSocket for Real-Time Updates

```python
import websockets
import json

async def monitor_websocket(batch_id: str):
    """Monitor progress via WebSocket."""
    uri = f"ws://localhost:8000/ws/upload/{batch_id}"

    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)

            if data["type"] == "progress":
                print(f"Progress: {data['processed']}/{data['total']}")
            elif data["type"] == "document_complete":
                print(f"Completed: {data['filename']}")
            elif data["type"] == "batch_complete":
                print("Batch processing complete!")
                break
```

## Step 4: Using Ray for ML Workloads

For heavy ML workloads (embeddings, KG extraction), Ray provides better performance:

```python
from backend.services.distributed_processor import get_distributed_processor

async def process_with_ray():
    """Process documents using Ray."""
    processor = await get_distributed_processor()

    # Check if Ray is available
    health = await processor.health_check()
    print(f"Ray available: {health['backends']['ray']['available']}")

    # Process embeddings (automatically uses Ray if available)
    texts = ["Document 1 content...", "Document 2 content..."]
    embeddings = await processor.process_embeddings(texts)

    # Extract knowledge graphs in parallel
    documents = [
        {"id": "doc1", "content": "..."},
        {"id": "doc2", "content": "..."},
    ]
    kg_results = await processor.extract_knowledge_graph(documents)
```

## Configuration Options

### Environment Variables

```env
# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_WORKER_CONCURRENCY=4
BULK_UPLOAD_MAX_CONCURRENT=4
BULK_UPLOAD_BATCH_SIZE=100

# Ray Configuration
RAY_ADDRESS=auto
RAY_NUM_WORKERS=8
USE_RAY_FOR_EMBEDDINGS=true
USE_RAY_FOR_KG=true
```

### Priority Queues

AIDocumentIndexer uses 5 priority queues:

| Queue | Priority | Use Case |
|-------|----------|----------|
| critical | 10 | User chat/search |
| high | 7 | Audio preview, quick queries |
| default | 5 | Standard processing |
| batch | 3 | Bulk uploads |
| background | 1 | KG extraction, analytics |

## Performance Optimization

### 1. Increase Worker Count

```bash
# More Celery workers
celery -A backend.services.task_queue worker -l info -c 8

# More Ray workers
export RAY_NUM_WORKERS=16
```

### 2. Use GPU for Embeddings

```env
ENABLE_GPU_EMBEDDINGS=true
```

### 3. Adjust Batch Sizes

```env
EMBEDDING_BATCH_SIZE=200  # Larger batches for GPU
KG_EXTRACTION_CONCURRENCY=8  # More parallel extractions
```

## Troubleshooting

### Documents Stuck in Queue

```bash
# Check Celery queue status
celery -A backend.services.task_queue inspect active

# Check Redis for pending tasks
redis-cli LLEN celery
```

### Memory Issues

```env
# Reduce concurrency
CELERY_WORKER_CONCURRENCY=2
BULK_UPLOAD_MAX_CONCURRENT=2

# Enable memory-mapped indexes
COLBERT_MMAP_ENABLED=true
```

### Ray Connection Issues

```bash
# Check Ray cluster status
ray status

# Restart Ray
ray stop
ray start --head --num-cpus=8
```

## Next Steps

- [Ray Scaling Tutorial](10-ray-scaling.md) - Advanced Ray configuration
- [Knowledge Graph Tutorial](05-knowledge-graph.md) - Explore extracted entities
- [Advanced RAG Tutorial](11-advanced-rag.md) - Improve retrieval quality
