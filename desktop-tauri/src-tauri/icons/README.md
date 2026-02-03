# App Icons

This directory should contain the following icon files for the Tauri desktop application:

## Required Icons

### macOS
- `icon.icns` - macOS app icon (contains multiple sizes)

### Windows
- `icon.ico` - Windows app icon (contains multiple sizes)

### Linux
- `32x32.png` - 32x32 pixels
- `128x128.png` - 128x128 pixels
- `128x128@2x.png` - 256x256 pixels (HiDPI)
- `icon.png` - 512x512 pixels (default)

## Generating Icons

You can use the Tauri CLI to generate icons from a source PNG (1024x1024 recommended):

```bash
# From the desktop-tauri directory
npx tauri icon public/icon.png
```

Or manually using ImageMagick:

```bash
# Generate PNG sizes
convert icon.svg -resize 32x32 icons/32x32.png
convert icon.svg -resize 128x128 icons/128x128.png
convert icon.svg -resize 256x256 icons/128x128@2x.png
convert icon.svg -resize 512x512 icons/icon.png

# Generate ICO for Windows
convert icon.svg -define icon:auto-resize=256,128,64,48,32,16 icons/icon.ico

# Generate ICNS for macOS (requires iconutil on macOS)
mkdir icon.iconset
convert icon.svg -resize 16x16 icon.iconset/icon_16x16.png
convert icon.svg -resize 32x32 icon.iconset/icon_16x16@2x.png
convert icon.svg -resize 32x32 icon.iconset/icon_32x32.png
convert icon.svg -resize 64x64 icon.iconset/icon_32x32@2x.png
convert icon.svg -resize 128x128 icon.iconset/icon_128x128.png
convert icon.svg -resize 256x256 icon.iconset/icon_128x128@2x.png
convert icon.svg -resize 256x256 icon.iconset/icon_256x256.png
convert icon.svg -resize 512x512 icon.iconset/icon_256x256@2x.png
convert icon.svg -resize 512x512 icon.iconset/icon_512x512.png
convert icon.svg -resize 1024x1024 icon.iconset/icon_512x512@2x.png
iconutil -c icns icon.iconset -o icons/icon.icns
```
