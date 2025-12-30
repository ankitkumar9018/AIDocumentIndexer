# OCR Configuration Guide

Complete guide to configuring and managing OCR (Optical Character Recognition) in AIDocumentIndexer.

## Table of Contents

- [Overview](#overview)
- [OCR Providers](#ocr-providers)
- [Configuration via Admin UI](#configuration-via-admin-ui)
- [Configuration via Settings](#configuration-via-settings)
- [Model Management](#model-management)
- [Supported Languages](#supported-languages)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [OCR Metrics & Analytics](#ocr-metrics--analytics-new) (New!)
- [Migration from Environment Variables](#migration-from-environment-variables)

---

## Overview

AIDocumentIndexer supports multiple OCR engines for extracting text from images and scanned documents:

- **PaddleOCR**: Deep learning-based OCR with high accuracy (recommended)
- **EasyOCR**: PyTorch-based OCR with GPU optimization (new!)
- **Tesseract**: Traditional OCR engine, fast and lightweight
- **Auto Mode**: Tries multiple engines with automatic fallback

All OCR configuration is managed through the database-backed settings system, accessible via the Admin UI or API.

**New Features:**
- 🎯 **Performance Monitoring**: Track OCR metrics and analytics
- 📦 **Batch Model Downloads**: Download multiple languages efficiently
- 🔄 **Auto-Update Checking**: Check for new model versions
- 🚀 **EasyOCR Support**: GPU-accelerated alternative OCR engine

---

## OCR Providers

### PaddleOCR (Recommended)

**Advantages:**
- Higher accuracy for complex documents
- Multi-language support with dedicated models
- Text orientation detection
- Better handling of rotated/skewed text

**Disadvantages:**
- Larger model downloads (100+ MB)
- Slower processing than Tesseract
- Requires model downloads on first use

**Model Variants:**
- `server`: High accuracy, slower processing (~2-3s per page)
- `mobile`: Fast processing, lower accuracy (~0.5-1s per page)

### Tesseract

**Advantages:**
- Fast processing
- Lightweight (no model downloads)
- Wide language support via system packages

**Disadvantages:**
- Lower accuracy on complex documents
- Poor handling of rotated/skewed text
- Requires system installation

### EasyOCR (New!)

**Advantages:**
- GPU-accelerated processing (2-3x faster than PaddleOCR on GPU)
- PyTorch-based, modern architecture
- Excellent for Asian languages
- Automatic CPU fallback

**Disadvantages:**
- Requires GPU for best performance
- Higher memory usage (~1-2GB GPU memory)
- Longer initialization time

**Best For:**
- Environments with GPU available
- Asian language documents
- High-throughput processing

### Auto Mode

Combines multiple engines:
1. Attempts PaddleOCR first for high accuracy
2. Falls back to EasyOCR if available
3. Falls back to Tesseract as last resort
4. Configurable via `ocr.tesseract.fallback_enabled` setting

---

## Configuration via Admin UI

### Accessing OCR Settings

1. Navigate to **Dashboard** → **Admin** → **Settings**
2. Click the **OCR Configuration** tab
3. Configure settings and save

### Available Settings

#### OCR Provider
- **PaddleOCR**: Deep learning-based (high accuracy)
- **EasyOCR**: PyTorch-based (GPU-optimized)
- **Tesseract**: Traditional OCR (fast)
- **Auto**: Try all with fallback chain

#### Model Variant (PaddleOCR only)
- **Server**: Accurate but slower
- **Mobile**: Fast but less accurate

#### Languages
Select from supported languages:
- English (en)
- German (de)
- French (fr)
- Spanish (es)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)

**Note**: Each language requires a separate model download (~30-50 MB per language).

#### Auto-Download Models
When enabled, missing models are automatically downloaded on:
- Application startup
- Settings change (new language added)
- First OCR operation

#### Tesseract Fallback
Enable automatic fallback to Tesseract if PaddleOCR fails.

---

## Configuration via Settings

### Database Settings

All OCR configuration is stored in the `system_settings` table:

| Setting Key | Type | Default | Description |
|------------|------|---------|-------------|
| `ocr.provider` | string | `paddleocr` | OCR engine selection |
| `ocr.paddle.variant` | string | `server` | Model variant |
| `ocr.paddle.languages` | json | `["en", "de"]` | Language codes |
| `ocr.paddle.model_dir` | string | `./data/paddle_models` | Model storage path |
| `ocr.paddle.auto_download` | boolean | `true` | Auto-download on startup |
| `ocr.easyocr.languages` | json | `["en"]` | EasyOCR language codes |
| `ocr.easyocr.use_gpu` | boolean | `true` | Use GPU if available |
| `ocr.tesseract.fallback_enabled` | boolean | `true` | Tesseract fallback |

### Environment Variables

For backward compatibility, environment variables are still supported:

```bash
# PaddleOCR model directory
PADDLEX_HOME=./data/paddle_models
PADDLE_HUB_HOME=./data/paddle_models/official_models

# Use HuggingFace mirror for faster downloads
PADDLE_PDX_MODEL_SOURCE=HF
```

**Note**: Database settings take precedence over environment variables.

---

## Model Management

### Model Storage

PaddleOCR models are stored in: `./data/paddle_models/`

**Directory Structure:**
```
data/paddle_models/
├── official_models/
│   ├── PP-OCRv5_server_det/        # Text detection model
│   ├── PP-LCNet_x1_0_textline_ori/ # Text orientation model
│   └── latin_PP-OCRv5_mobile_rec/  # Recognition model (Latin)
├── func_ret/
├── locks/
└── temp/
```

### Model Downloads

**Via Admin UI:**
1. Go to OCR Configuration tab
2. Select languages and variant
3. Click "Download Selected Models"
4. Wait for download completion

**Via Script:**
```bash
python backend/scripts/download_paddle_models.py
```

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/admin/ocr/models/download \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"languages": ["en", "de"], "variant": "server"}'
```

### Model Migration

To migrate existing models from system location (`~/.paddlex`) to project directory:

```bash
python backend/scripts/migrate_paddle_models.py
```

This script:
- Copies models from `~/.paddlex` to `./data/paddle_models`
- Preserves existing models
- Shows size verification
- Provides cleanup instructions

### Docker Volumes

For Docker deployments, models are persisted via named volume:

```yaml
volumes:
  paddle_models:
    name: aidoc-paddle-models
```

This ensures models survive container rebuilds.

---

## Supported Languages

### PaddleOCR Languages

PaddleOCR supports 80+ languages. Common languages available in the UI:

| Code | Language | Model Size |
|------|----------|------------|
| `en` | English | ~40 MB |
| `de` | German | ~40 MB |
| `fr` | French | ~40 MB |
| `es` | Spanish | ~40 MB |
| `zh` | Chinese | ~50 MB |
| `ja` | Japanese | ~45 MB |
| `ko` | Korean | ~45 MB |
| `ar` | Arabic | ~45 MB |

**Full language list**: [PaddleOCR Language Codes](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)

### Tesseract Languages

Tesseract supports 100+ languages. Install via system package manager:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr-deu tesseract-ocr-fra

# macOS
brew install tesseract-lang
```

---

## Performance Tuning

### Choosing the Right Provider

**Use PaddleOCR Server when:**
- Accuracy is critical
- Processing complex documents (tables, multi-column)
- Handling rotated/skewed images
- Processing time is not critical

**Use PaddleOCR Mobile when:**
- Fast processing is needed
- Simple documents (clean scans)
- Moderate accuracy is acceptable

**Use Tesseract when:**
- Fastest processing required
- Lightweight deployment needed
- Simple document layouts

**Use Auto Mode when:**
- Need both accuracy and reliability
- Acceptable to trade speed for quality
- Mixed document quality

### Model Preloading

Models are lazy-loaded by default. To preload on startup, ensure `ocr.paddle.auto_download` is enabled.

### Caching

OCR results are cached at the document level. Re-processing the same document retrieves cached text.

---

## Troubleshooting

### Models Not Downloading

**Symptom**: "No valid PaddlePaddle model found" error

**Solutions:**
1. Check internet connection
2. Verify `PADDLE_PDX_MODEL_SOURCE=HF` is set
3. Check disk space (models need ~100-200 MB)
4. Try manual download:
   ```bash
   python backend/scripts/download_paddle_models.py
   ```

### PaddleOCR Initialization Fails

**Symptom**: OCR falls back to Tesseract unexpectedly

**Solutions:**
1. Check model directory exists: `ls -lh data/paddle_models/`
2. Verify settings: `GET /api/v1/admin/ocr/settings`
3. Check logs for specific errors
4. Ensure correct Python dependencies:
   ```bash
   pip install paddleocr paddlepaddle
   ```

### Tesseract Not Found

**Symptom**: "Tesseract not found or not working" error

**Solutions:**
1. Install Tesseract:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr

   # macOS
   brew install tesseract
   ```
2. Verify installation:
   ```bash
   tesseract --version
   ```

### Poor OCR Accuracy

**Solutions:**
1. Switch to PaddleOCR Server variant
2. Ensure correct language is selected
3. Check image quality (resolution, contrast)
4. Try manual preprocessing (deskew, denoise)

### Slow Processing

**Solutions:**
1. Switch to PaddleOCR Mobile variant
2. Use Tesseract for simple documents
3. Enable caching to avoid re-processing
4. Consider GPU acceleration (requires CUDA)

---

## API Reference

### Get OCR Settings

```http
GET /api/v1/admin/ocr/settings
Authorization: Bearer {token}
```

**Response:**
```json
{
  "settings": {
    "ocr.provider": "paddleocr",
    "ocr.paddle.variant": "server",
    "ocr.paddle.languages": ["en", "de"],
    "ocr.paddle.model_dir": "./data/paddle_models",
    "ocr.paddle.auto_download": true,
    "ocr.tesseract.fallback_enabled": true
  },
  "models": {
    "downloaded": [
      {
        "name": "inference",
        "type": ".pdiparams",
        "size": "83.9 MB",
        "path": "official_models/PP-OCRv5_server_det/inference.pdiparams"
      }
    ],
    "total_size": "118.4 MB",
    "model_dir": "data/paddle_models",
    "status": "installed"
  }
}
```

### Update OCR Settings

```http
PATCH /api/v1/admin/ocr/settings
Authorization: Bearer {token}
Content-Type: application/json

{
  "ocr.provider": "paddleocr",
  "ocr.paddle.variant": "mobile",
  "ocr.paddle.languages": ["en", "de", "fr"]
}
```

**Response:**
```json
{
  "status": "updated",
  "settings": { ... },
  "download_triggered": true
}
```

### Download Models

```http
POST /api/v1/admin/ocr/models/download
Authorization: Bearer {token}
Content-Type: application/json

{
  "languages": ["en", "de"],
  "variant": "server"
}
```

**Response:**
```json
{
  "status": "success",
  "downloaded": ["en", "de"],
  "failed": [],
  "model_info": { ... }
}
```

### Get Model Info

```http
GET /api/v1/admin/ocr/models/info
Authorization: Bearer {token}
```

**Response:**
```json
{
  "downloaded": [ ... ],
  "total_size": "118.4 MB",
  "model_dir": "data/paddle_models",
  "status": "installed"
}
```

### Batch Download Models (New!)

```http
POST /api/v1/admin/ocr/models/download-batch
Authorization: Bearer {token}
Content-Type: application/json

{
  "language_batches": [["en", "de"], ["fr", "es"]],
  "variant": "server"
}
```

**Response:**
```json
{
  "status": "success",
  "downloaded": ["en", "de", "fr", "es"],
  "failed": [],
  "model_info": { ... },
  "batches_processed": 2
}
```

**Use Case:** Download many languages efficiently without blocking for too long.

### Check Model Updates (New!)

```http
GET /api/v1/admin/ocr/models/check-updates
Authorization: Bearer {token}
```

**Response:**
```json
{
  "status": "success",
  "current_version": "2.7.0",
  "latest_version": "2.8.0",
  "update_available": true,
  "release_date": "2025-12-15T10:30:00"
}
```

### Get Installed Models Info (New!)

```http
GET /api/v1/admin/ocr/models/installed
Authorization: Bearer {token}
```

**Response:**
```json
{
  "status": "success",
  "models": [
    {
      "name": "PP-OCRv5_server_det",
      "path": "official_models/PP-OCRv5_server_det",
      "size_mb": 83.9,
      "last_modified": "2025-12-30T10:15:00"
    }
  ],
  "total_models": 5,
  "model_dir": "data/paddle_models"
}
```

---

## OCR Metrics & Analytics (New!)

Track and analyze OCR performance with comprehensive metrics.

### Get Metrics Summary

```http
GET /api/v1/admin/ocr/metrics/summary?days=7&provider=paddleocr
Authorization: Bearer {token}
```

**Response:**
```json
{
  "total_operations": 1250,
  "successful_operations": 1200,
  "success_rate": 96.0,
  "average_processing_time_ms": 2450,
  "total_characters_processed": 125000,
  "total_cost_usd": 0.0,
  "fallback_used_count": 50,
  "fallback_usage_rate": 4.0
}
```

### Get Metrics by Provider

```http
GET /api/v1/admin/ocr/metrics/by-provider?days=30
Authorization: Bearer {token}
```

**Response:**
```json
[
  {
    "provider": "paddleocr",
    "total_operations": 800,
    "success_rate": 97.5,
    "average_processing_time_ms": 2300
  },
  {
    "provider": "easyocr",
    "total_operations": 200,
    "success_rate": 95.0,
    "average_processing_time_ms": 1500
  }
]
```

### Get Metrics by Language

```http
GET /api/v1/admin/ocr/metrics/by-language?days=30
Authorization: Bearer {token}
```

**Response:**
```json
[
  {
    "language": "en",
    "total_operations": 600,
    "average_processing_time_ms": 2200
  },
  {
    "language": "de",
    "total_operations": 400,
    "average_processing_time_ms": 2400
  }
]
```

### Get Performance Trend

```http
GET /api/v1/admin/ocr/metrics/trend?days=14&provider=paddleocr
Authorization: Bearer {token}
```

**Response:**
```json
[
  {
    "date": "2025-12-16",
    "total_operations": 85,
    "success_rate": 96.5,
    "average_processing_time_ms": 2350
  },
  {
    "date": "2025-12-17",
    "total_operations": 92,
    "success_rate": 97.8,
    "average_processing_time_ms": 2280
  }
]
```

### Get Recent Operations

```http
GET /api/v1/admin/ocr/metrics/recent?limit=50&provider=paddleocr
Authorization: Bearer {token}
```

**Response:**
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "provider": "paddleocr",
    "language": "en",
    "processing_time_ms": 2150,
    "success": true,
    "character_count": 1250,
    "created_at": "2025-12-30T14:23:15Z"
  }
]
```

---

## Migration from Environment Variables

If you previously configured OCR via environment variables, migrate to database settings:

1. **Backup Current Settings:**
   ```bash
   # Note your current OCR_LANGUAGE and other env vars
   env | grep -E 'OCR|PADDLE'
   ```

2. **Run Settings Migration:**
   ```bash
   python backend/scripts/init_ocr_settings.py
   ```

3. **Migrate Models:**
   ```bash
   python backend/scripts/migrate_paddle_models.py
   ```

4. **Update Configuration:**
   - Use Admin UI to set provider, variant, and languages
   - Or update via API

5. **Remove Environment Variables:**
   - Update `.env` file
   - Restart application

---

## Best Practices

1. **Start with Auto Mode**: Provides best balance of accuracy and reliability
2. **Use Server Variant for Production**: Higher quality results worth the processing time
3. **Pre-download Models**: Enable auto-download to avoid delays during processing
4. **Monitor Model Storage**: Models can consume 100-500 MB depending on languages
5. **Use Docker Volumes**: Persist models across container rebuilds
6. **Enable Tesseract Fallback**: Ensures OCR always succeeds
7. **Test with Sample Documents**: Verify OCR quality before production use
8. **Use Batch Downloads**: Download multiple languages efficiently with batch API
9. **Monitor Performance**: Leverage metrics endpoints to track OCR performance
10. **Check for Updates**: Periodically check for new model versions
11. **Consider EasyOCR for GPU**: If GPU available, EasyOCR can provide faster processing
12. **Review Metrics Regularly**: Use analytics to optimize provider selection

---

## What's New

### Version 2.0 (2025-12-30)

**New Features:**
- ✅ **EasyOCR Provider**: GPU-accelerated OCR alternative
- ✅ **Performance Metrics**: Comprehensive analytics and monitoring
- ✅ **Batch Downloads**: Efficient multi-language model downloads
- ✅ **Auto-Update Checking**: Check for new PaddleOCR versions
- ✅ **Enhanced API**: 8 new endpoints for metrics and model management

**New Endpoints:**
- `POST /api/v1/admin/ocr/models/download-batch` - Batch model downloads
- `GET /api/v1/admin/ocr/models/check-updates` - Check for updates
- `GET /api/v1/admin/ocr/models/installed` - Installed models info
- `GET /api/v1/admin/ocr/metrics/summary` - Metrics summary
- `GET /api/v1/admin/ocr/metrics/by-provider` - Provider comparison
- `GET /api/v1/admin/ocr/metrics/by-language` - Language statistics
- `GET /api/v1/admin/ocr/metrics/trend` - Performance trends
- `GET /api/v1/admin/ocr/metrics/recent` - Recent operations

**New Settings:**
- `ocr.easyocr.languages` - EasyOCR language configuration
- `ocr.easyocr.use_gpu` - GPU acceleration toggle

---

## Support

For issues or questions:
- Check [GitHub Issues](https://github.com/your-repo/issues)
- Review [Troubleshooting](#troubleshooting) section
- Check application logs for detailed errors
- Test API endpoints directly for debugging
- See [OCR_ENHANCEMENTS_SUMMARY.md](./OCR_ENHANCEMENTS_SUMMARY.md) for implementation details

---

**Last Updated**: 2025-12-30
**Version**: 2.0.0
**Major Updates**: EasyOCR support, Performance metrics, Batch downloads, Auto-update checking
