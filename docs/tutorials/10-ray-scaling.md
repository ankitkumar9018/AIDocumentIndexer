# Tutorial: Horizontal Scaling with Ray

Learn how to scale AIDocumentIndexer horizontally using Ray for distributed ML workloads.

## Overview

Ray is used alongside Celery for heavy ML workloads:

| Backend | Best For | Tasks |
|---------|----------|-------|
| Celery | Async I/O | File uploads, notifications, cleanup |
| Ray | ML Compute | Embeddings, KG extraction, VLM processing |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  API Server  │ │  API Server  │ │  API Server  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌─────────────────┐
              │ Redis (Broker)  │
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Celery Worker│ │Celery Worker│ │Celery Worker│
└─────────────┘ └─────────────┘ └─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Ray Cluster                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │Ray Head │  │Ray Worker│  │Ray Worker│  │Ray Worker│          │
│  │(Control)│  │  (GPU)   │  │  (GPU)   │  │  (CPU)   │          │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Start Ray Cluster

### Local Development

```bash
# Start Ray head node
ray start --head --num-cpus=8 --num-gpus=1

# Check status
ray status
```

### Multi-Node Cluster

On the head node:
```bash
ray start --head --port=6379
```

On worker nodes:
```bash
ray start --address='<head-ip>:6379'
```

## Step 2: Configure AIDocumentIndexer

```env
# Ray connection
RAY_ADDRESS=auto  # or ray://<head-ip>:10001

# Worker configuration
RAY_NUM_WORKERS=8

# Task routing
USE_RAY_FOR_EMBEDDINGS=true
USE_RAY_FOR_KG=true
USE_RAY_FOR_VLM=true
```

## Step 3: Using the Distributed Processor

```python
from backend.services.distributed_processor import get_distributed_processor

async def process_documents():
    """Process documents with automatic Ray/Celery routing."""
    processor = await get_distributed_processor()

    # Check health
    health = await processor.health_check()
    print(f"Ray status: {health['backends']['ray']}")
    print(f"Active pools: {health['pools']}")

    # Process embeddings (uses Ray ActorPool)
    texts = ["doc1 content", "doc2 content", ...]
    embeddings = await processor.process_embeddings(
        texts,
        batch_size=100  # Process 100 texts per batch
    )

    # Extract knowledge graphs (uses Ray ActorPool)
    documents = [
        {"id": "doc1", "content": "..."},
        {"id": "doc2", "content": "..."},
    ]
    kg_results = await processor.extract_knowledge_graph(
        documents,
        batch_size=10
    )

    # Process visual documents (uses Ray ActorPool)
    visual_docs = [
        {"id": "img1", "image_data": b"..."},
        {"id": "img2", "image_data": b"..."},
    ]
    vlm_results = await processor.process_visual_documents(visual_docs)
```

## Step 4: Custom Ray Tasks

For custom ML workloads:

```python
from backend.services.ray_cluster import get_ray_manager

async def custom_ml_task():
    """Run a custom ML task on Ray."""
    manager = await get_ray_manager()

    if not manager.is_available:
        print("Ray not available, running locally")
        return my_ml_function(data)

    # Submit single task
    result = await manager.submit_task(
        my_ml_function,
        data,
        num_cpus=2,
        num_gpus=0.5,
    )

    # Submit batch of tasks
    batch_args = [
        ((data1,), {}),
        ((data2,), {}),
        ((data3,), {}),
    ]
    results = await manager.submit_batch(
        my_ml_function,
        batch_args,
        num_cpus=1,
    )

    return results
```

## Step 5: Actor Pools

For stateful workloads:

```python
from backend.services.ray_cluster import get_ray_manager

class MyMLActor:
    def __init__(self, model_name):
        self.model = load_model(model_name)

    def predict(self, inputs):
        return self.model(inputs)

async def use_actor_pool():
    """Use an actor pool for stateful processing."""
    manager = await get_ray_manager()

    # Create actor pool
    pool = await manager.get_actor_pool(
        name="my_ml_pool",
        actor_class=MyMLActor,
        pool_size=4,
        "gpt-4o",  # model_name arg
    )

    if pool is None:
        print("Could not create pool")
        return

    # Process with pool
    inputs = [data1, data2, data3, data4]
    results = list(pool.map(
        lambda actor, x: actor.predict.remote(x),
        inputs
    ))
```

## Kubernetes Deployment

### Ray Operator

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: aidocindexer-ray
spec:
  rayVersion: '2.10.0'
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      spec:
        containers:
        - name: ray-head
          image: rayproject/ray:2.10.0-py311
          resources:
            limits:
              cpu: "4"
              memory: "8Gi"
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 4
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray:2.10.0-py311-gpu
          resources:
            limits:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: 1
```

### Application Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aidocindexer
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: aidocindexer:latest
        env:
        - name: RAY_ADDRESS
          value: "ray://aidocindexer-ray-head-svc:10001"
        - name: RAY_NUM_WORKERS
          value: "16"
        - name: USE_RAY_FOR_EMBEDDINGS
          value: "true"
```

## Monitoring

### Ray Dashboard

Access at `http://<ray-head>:8265`

Features:
- Cluster overview
- Node status
- Task metrics
- Memory usage
- Actor pools

### Health Check API

```python
# GET /api/health/ray
{
    "status": "healthy",
    "ray_status": "connected",
    "nodes": 5,
    "cpus": {"total": 40, "available": 32.5},
    "gpus": {"total": 4, "available": 2.0},
    "actor_pools": ["embedding_workers", "kg_workers", "vlm_workers"]
}
```

## Performance Tuning

### 1. Worker Scaling

```env
# Scale based on workload
RAY_NUM_WORKERS=16  # Increase for more parallelism
```

### 2. GPU Utilization

```python
# Request GPU resources
await manager.submit_task(
    gpu_function,
    data,
    num_gpus=1.0,  # Full GPU
)

# Or fractional GPUs for lightweight tasks
await manager.submit_task(
    lightweight_function,
    data,
    num_gpus=0.25,  # Share GPU with 3 other tasks
)
```

### 3. Memory Management

```python
# Use object store for large data
import ray

@ray.remote
def process_large_data(data_ref):
    data = ray.get(data_ref)
    return process(data)

# Put large data in object store
data_ref = ray.put(large_dataset)

# Pass reference (not data) to tasks
results = ray.get([
    process_large_data.remote(data_ref)
    for _ in range(10)
])
```

## Troubleshooting

### Ray Connection Failed

```bash
# Check Ray is running
ray status

# Check port connectivity
nc -zv <ray-head> 6379
nc -zv <ray-head> 10001
```

### Out of Memory

```env
# Reduce worker pool sizes
RAY_NUM_WORKERS=4

# Enable memory-mapped indexes
COLBERT_MMAP_ENABLED=true
```

### Tasks Not Scheduling

```bash
# Check available resources
ray status | grep -A 5 "Demands"

# Scale up workers
ray stop
ray start --head --num-cpus=16 --num-gpus=4
```

## Next Steps

- [Bulk Processing Tutorial](06-bulk-processing.md) - Process 100K+ documents
- [Advanced RAG Tutorial](11-advanced-rag.md) - Improve retrieval quality
- [Visual Documents Tutorial](12-visual-documents.md) - VLM processing
