# Browser Extension Icons

This directory should contain the following icon files for the browser extension:

- `icon-16.png` - 16x16 pixels (toolbar icon)
- `icon-32.png` - 32x32 pixels (toolbar icon @2x)
- `icon-48.png` - 48x48 pixels (extension management page)
- `icon-128.png` - 128x128 pixels (Chrome Web Store)

## Generating Icons

You can generate these icons from an SVG source using ImageMagick:

```bash
# From the browser-extension/assets directory
convert icon.svg -resize 16x16 icons/icon-16.png
convert icon.svg -resize 32x32 icons/icon-32.png
convert icon.svg -resize 48x48 icons/icon-48.png
convert icon.svg -resize 128x128 icons/icon-128.png
```

Or use an online tool like https://realfavicongenerator.net/

## Design Guidelines

- Use a simple, recognizable design
- Works well at small sizes (16x16)
- Has good contrast on both light and dark backgrounds
- Follows Chrome Web Store icon guidelines
