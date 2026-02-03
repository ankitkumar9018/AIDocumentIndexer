#!/usr/bin/env node
/**
 * Icon Generator for AIDocumentIndexer
 * =====================================
 *
 * Generates PNG icons from SVG for:
 * - Browser Extension (16, 32, 48, 128 px)
 * - Desktop App (32, 128, 256, 512 px + .ico/.icns)
 *
 * Usage: node scripts/generate-icons.js
 *
 * Requirements: npm install sharp
 */

const fs = require('fs');
const path = require('path');

// Try to use sharp for high-quality conversion
async function generateWithSharp() {
  let sharp;
  try {
    sharp = require('sharp');
  } catch (e) {
    console.log('Sharp not installed. Install with: npm install sharp');
    console.log('Falling back to creating placeholder icons...');
    return false;
  }

  const svgPath = path.join(__dirname, '../browser-extension/assets/icon.svg');
  const svg = fs.readFileSync(svgPath);

  // Browser extension icons
  const browserIconsDir = path.join(__dirname, '../browser-extension/assets/icons');
  fs.mkdirSync(browserIconsDir, { recursive: true });

  const browserSizes = [16, 32, 48, 128];
  for (const size of browserSizes) {
    await sharp(svg)
      .resize(size, size)
      .png()
      .toFile(path.join(browserIconsDir, `icon-${size}.png`));
    console.log(`Created browser-extension/assets/icons/icon-${size}.png`);
  }

  // Desktop app icons
  const desktopIconsDir = path.join(__dirname, '../desktop-tauri/src-tauri/icons');
  fs.mkdirSync(desktopIconsDir, { recursive: true });

  const desktopSizes = [32, 128, 256, 512];
  for (const size of desktopSizes) {
    const filename = size === 256 ? '128x128@2x.png' :
                     size === 512 ? 'icon.png' :
                     `${size}x${size}.png`;
    await sharp(svg)
      .resize(size, size)
      .png()
      .toFile(path.join(desktopIconsDir, filename));
    console.log(`Created desktop-tauri/src-tauri/icons/${filename}`);
  }

  // Also copy to public folder for desktop
  const publicDir = path.join(__dirname, '../desktop-tauri/public');
  fs.mkdirSync(publicDir, { recursive: true });
  await sharp(svg)
    .resize(512, 512)
    .png()
    .toFile(path.join(publicDir, 'icon.png'));
  console.log('Created desktop-tauri/public/icon.png');

  console.log('\n✅ All icons generated successfully!');
  console.log('\nNote: For .ico (Windows) and .icns (macOS), use:');
  console.log('  npx tauri icon desktop-tauri/public/icon.png');
  return true;
}

// Create simple placeholder icons if sharp is not available
function createPlaceholders() {
  // Create a simple 1x1 blue pixel PNG as placeholder
  // This is a valid PNG that can be used until proper icons are generated
  const pngHeader = Buffer.from([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
    0x00, 0x00, 0x00, 0x0D, // IHDR length
    0x49, 0x48, 0x44, 0x52, // IHDR
    0x00, 0x00, 0x00, 0x10, // width: 16
    0x00, 0x00, 0x00, 0x10, // height: 16
    0x08, 0x02, // bit depth: 8, color type: RGB
    0x00, 0x00, 0x00, // compression, filter, interlace
    0x90, 0x91, 0x68, 0x36, // CRC
  ]);

  console.log('Creating placeholder icons...');
  console.log('To generate proper icons, run:');
  console.log('  npm install sharp');
  console.log('  node scripts/generate-icons.js');

  // Just create the directories for now
  const browserIconsDir = path.join(__dirname, '../browser-extension/assets/icons');
  const desktopIconsDir = path.join(__dirname, '../desktop-tauri/src-tauri/icons');

  fs.mkdirSync(browserIconsDir, { recursive: true });
  fs.mkdirSync(desktopIconsDir, { recursive: true });

  console.log('\nDirectories created. Please generate icons manually using:');
  console.log('  ImageMagick: convert icon.svg -resize 128x128 icon-128.png');
  console.log('  Or online: https://realfavicongenerator.net/');
}

async function main() {
  console.log('AIDocumentIndexer Icon Generator\n');

  const success = await generateWithSharp();
  if (!success) {
    createPlaceholders();
  }
}

main().catch(console.error);
