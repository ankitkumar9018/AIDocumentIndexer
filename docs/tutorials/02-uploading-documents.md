# Uploading Documents

Learn how to upload and manage documents in AIDocumentIndexer.

## Supported File Types

- **Documents**: PDF, DOCX, DOC, TXT, MD, RTF
- **Spreadsheets**: XLSX, XLS, CSV
- **Presentations**: PPTX, PPT
- **Images**: PNG, JPG, TIFF (with OCR)
- **Email**: MSG, EML

## Upload Methods

### Single File Upload

1. Click the "Upload" button in the sidebar
2. Select a file from your computer
3. Choose the target collection (optional)
4. Click "Upload"

### Bulk Upload

For large document sets, see [Bulk Processing](06-bulk-processing.md).

### Web Scraping

Import content from websites using the web scraper feature.

## Processing Options

### OCR for Scanned Documents

Enable OCR in the upload settings for scanned PDFs and images.

### Chunking Strategy

- **Recursive**: Default, works well for most documents
- **Semantic**: Better for structured documents with clear sections
- **Fast (Chonkie)**: 33x faster, enable via `ENABLE_FAST_CHUNKING=true`

## Metadata and Collections

Organize documents with:
- **Collections**: Group related documents
- **Tags**: Add searchable tags
- **Access Tiers**: Control visibility

## Next Steps

- [Chat Interface](03-chat-interface.md) - Query your uploaded documents
