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

### Auto-Generate Tags

Enable automatic tag generation to have AI analyze your document and suggest relevant tags.

1. Toggle "Auto-Generate Tags" in Processing Options before uploading
2. After processing, the system analyzes the document name and content
3. Up to 5 relevant tags are generated and applied to the document
4. Tags focus on: topic, industry, document type, and project names

**Example**: A project management guide might get tags like: `Project Management`, `Software Development`, `Guide`, `Best Practices`

The LLM used for tag generation is configurable in Admin Settings under "LLM Configuration" > "Operation-Level Config" > "auto_tagging".

### Duplicate Detection

By default, the system detects duplicate uploads based on file content hash:

- If you upload a file that already exists, you'll see a "Duplicate file detected" notification
- The upload is skipped, and you're shown the existing document
- Re-uploading a previously **deleted** file is allowed (treated as new)

To disable duplicate detection for a specific upload, uncheck "Detect Duplicates" in Processing Options.

## Metadata and Collections

Organize documents with:
- **Collections**: Group related documents
- **Tags**: Add searchable tags
- **Access Tiers**: Control visibility

## Next Steps

- [Chat Interface](03-chat-interface.md) - Query your uploaded documents
