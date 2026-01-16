# Document Templates

This folder contains default templates for document generation.

## Template Structure

```
templates/
├── pptx/           # PowerPoint templates
│   ├── corporate/  # Professional business templates
│   ├── creative/   # Bold, colorful designs
│   ├── academic/   # Research and education
│   └── pitch/      # Startup and investor decks
│
├── docx/           # Word document templates
│   ├── reports/    # Business and technical reports
│   ├── proposals/  # Project and business proposals
│   └── letters/    # Formal letters and memos
│
├── pdf/            # HTML/CSS templates for PDF generation
│   ├── reports/    # HTML templates for PDF reports
│   └── stylesheets/# CSS for PDF styling
│
└── xlsx/           # Excel spreadsheet templates
    ├── financial/  # Budget, P&L, invoices
    ├── project/    # Project tracking
    └── data/       # Data analysis templates
```

## Adding Templates

1. Place your template file in the appropriate folder
2. Create a `template.json` manifest file with metadata:

```json
{
  "id": "unique-id",
  "name": "Display Name",
  "description": "Brief description",
  "category": "corporate",
  "file_type": "pptx",
  "preview_image": "preview.png",
  "tags": ["tag1", "tag2"],
  "primary_color": "#2563EB",
  "style": "minimal",
  "tone": "professional"
}
```

## External Template Sources

For more templates:

### PPTX
- [Slidesgo](https://slidesgo.com/)
- [SlidesCarnival](https://www.slidescarnival.com/)
- [PresentationGO](https://www.presentationgo.com/)

### DOCX
- [Microsoft Templates](https://word.cloud.microsoft/create/en/templates/)
- [Template.net](https://www.template.net/editable/word)

### XLSX
- [Vertex42](https://www.vertex42.com/ExcelTemplates/)
- [Microsoft Excel](https://excel.cloud.microsoft/create/en/templates/)

### PDF/HTML
- Use WeasyPrint + Jinja2 for custom HTML→PDF rendering
- CSS in `stylesheets/` folder for styling
