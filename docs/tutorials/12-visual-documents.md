# Tutorial: Processing Visual Documents

Learn how to process charts, tables, and images using AIDocumentIndexer's Vision Language Model (VLM) integration.

## Overview

AIDocumentIndexer supports multiple VLM providers for visual document understanding:

| Provider | Model | Best For |
|----------|-------|----------|
| Claude | claude-3-5-sonnet | Overall quality |
| OpenAI | gpt-4o | Speed |
| Qwen | Qwen3-VL | Local/privacy |
| Ollama | qwen3-vl | Self-hosted |

## Step 1: Configure VLM Provider

```env
# Choose provider
VLM_MODEL=claude  # claude, openai, qwen, local

# For Qwen local model
VLM_QWEN_MODEL=Qwen/Qwen3-VL-7B-Instruct

# Max images per request
VLM_MAX_IMAGES=10
```

## Step 2: Basic Image Analysis

```python
from backend.services.vlm_processor import get_vlm_processor

async def analyze_image():
    """Analyze an image with VLM."""
    processor = await get_vlm_processor()

    # From file path
    result = await processor.analyze_image(
        "/path/to/chart.png",
        prompt="Describe what this chart shows"
    )

    print(f"Analysis: {result.content}")
    print(f"Provider: {result.provider.value}")
    print(f"Processing time: {result.processing_time_ms}ms")

    # From bytes
    with open("/path/to/image.jpg", "rb") as f:
        image_bytes = f.read()

    result = await processor.analyze_image(
        image_bytes,
        prompt="What text is visible in this image?"
    )
```

## Step 3: Extract Text (OCR)

```python
async def extract_text_from_image():
    """Extract text from an image."""
    processor = await get_vlm_processor()

    result = await processor.extract_text("/path/to/document.png")

    print(f"Extracted text:\n{result.ocr_text}")

    # The text is also in result.content
    assert result.ocr_text == result.content
```

## Step 4: Analyze Charts

```python
async def analyze_chart():
    """Analyze a chart and extract data."""
    processor = await get_vlm_processor()

    result = await processor.describe_chart("/path/to/chart.png")

    if result.structured_data:
        data = result.structured_data
        print(f"Chart type: {data.get('chart_type')}")
        print(f"Title: {data.get('title')}")
        print(f"Data points: {data.get('data_points')}")
        print(f"Insights: {data.get('insights')}")
    else:
        # Fallback to text description
        print(f"Description: {result.content}")
```

## Step 5: Extract Tables

```python
async def extract_table():
    """Extract table data from an image."""
    processor = await get_vlm_processor()

    result = await processor.extract_table("/path/to/table.png")

    if result.structured_data:
        # Returns list of row objects
        for row in result.structured_data:
            print(row)
    else:
        print(f"Raw extraction: {result.content}")
```

## Step 6: Multi-Image Analysis

```python
async def analyze_multiple_images():
    """Analyze multiple images together."""
    processor = await get_vlm_processor()

    images = [
        "/path/to/page1.png",
        "/path/to/page2.png",
        "/path/to/page3.png",
    ]

    result = await processor.analyze_images(
        images,
        prompt="Compare these document pages and summarize the key differences",
        output_format="json"
    )

    if result.structured_data:
        print(f"Structured comparison: {result.structured_data}")
    else:
        print(f"Comparison: {result.content}")
```

## Step 7: Structured Output

Request JSON or HTML output:

```python
async def get_structured_output():
    """Get structured JSON output."""
    processor = await get_vlm_processor()

    # JSON output
    result = await processor.analyze_image(
        "/path/to/invoice.png",
        prompt="""Extract the following from this invoice:
        - Invoice number
        - Date
        - Total amount
        - Line items (description, quantity, price)""",
        output_format="json"
    )

    if result.structured_data:
        invoice = result.structured_data
        print(f"Invoice #: {invoice.get('invoice_number')}")
        print(f"Total: {invoice.get('total_amount')}")

    # HTML output (for formatted tables)
    result = await processor.analyze_image(
        "/path/to/table.png",
        prompt="Convert this table to HTML",
        output_format="html"
    )
    print(f"HTML:\n{result.content}")
```

## Step 8: Integration with Document Pipeline

Process visual documents during upload:

```python
from backend.services.vlm_processor import get_vlm_processor
from backend.services.document_parser import DocumentParser

async def process_visual_document(file_path: str):
    """Process a visual document with VLM integration."""
    # Check if document is visual
    parser = DocumentParser()
    doc_info = await parser.parse(file_path)

    if doc_info.is_visual_heavy:
        # Use VLM for better extraction
        vlm = await get_vlm_processor()

        # Extract text and structure
        pages_content = []
        for page_image in doc_info.page_images:
            result = await vlm.analyze_image(
                page_image,
                prompt="Extract all text, tables, and describe any charts or figures"
            )
            pages_content.append(result.content)

        doc_info.content = "\n\n".join(pages_content)

    return doc_info
```

## Using ColPali for Visual Retrieval

For visual document search (Phase 28):

```python
from backend.services.colpali_retriever import ColPaliRetriever

async def visual_search():
    """Search visual documents by content."""
    retriever = ColPaliRetriever()

    # Index visual documents
    await retriever.index_documents([
        "/path/to/doc1.pdf",
        "/path/to/doc2.pdf",
    ])

    # Search by query (finds relevant pages)
    results = await retriever.search(
        "quarterly revenue chart",
        top_k=5
    )

    for result in results:
        print(f"Document: {result.document_id}")
        print(f"Page: {result.page_number}")
        print(f"Score: {result.score}")
```

## Provider Fallback

VLM processor automatically falls back between providers:

```python
async def robust_analysis():
    """Analyze with automatic fallback."""
    processor = await get_vlm_processor()

    # If Claude fails, tries OpenAI automatically
    result = await processor.analyze_image(
        "/path/to/image.png",
        prompt="Describe this image"
    )

    print(f"Used provider: {result.provider.value}")
    print(f"Success: {result.success}")
```

## Health Check

```python
async def check_vlm_health():
    """Check VLM provider availability."""
    processor = await get_vlm_processor()

    health = await processor.health_check()
    print(f"Initialized: {health['initialized']}")
    print(f"Primary: {health['primary_provider']}")

    for provider, status in health['providers'].items():
        print(f"  {provider}: {'✓' if status['available'] else '✗'}")
```

## Performance Tips

### 1. Batch Processing

```python
# Process multiple images in one request (up to 10)
result = await processor.analyze_images(
    [img1, img2, img3, img4, img5],
    prompt="Summarize all these documents"
)
```

### 2. Use Local Models for Privacy

```env
VLM_MODEL=local
VLM_QWEN_MODEL=qwen3-vl  # via Ollama
```

### 3. Optimize Image Size

```python
from PIL import Image
import io

def optimize_image(image_bytes: bytes, max_size: int = 2048) -> bytes:
    """Resize image for faster processing."""
    img = Image.open(io.BytesIO(image_bytes))

    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
```

## Troubleshooting

### Provider Not Available

```python
# Check API keys
import os
print(f"Claude: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
print(f"OpenAI: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
```

### Slow Processing

```env
# Use faster provider
VLM_MODEL=openai  # GPT-4o is faster than Claude

# Or use local with GPU
VLM_MODEL=local
# Ensure Ollama has GPU access
```

### Out of Memory (Local Models)

```bash
# Use smaller model
ollama run qwen3-vl:3b  # Instead of 7b

# Or use cloud provider
VLM_MODEL=claude
```

## Next Steps

- [Advanced RAG Tutorial](11-advanced-rag.md) - Combine visual with text search
- [Bulk Processing Tutorial](06-bulk-processing.md) - Process many visual documents
- [Ray Scaling Tutorial](10-ray-scaling.md) - Distribute VLM workloads
