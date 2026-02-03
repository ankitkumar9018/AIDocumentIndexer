import { defineConfig } from 'vite';
import { resolve } from 'path';
import { copyFileSync, mkdirSync, existsSync } from 'fs';

/**
 * Vite config for Firefox extension build
 *
 * Firefox uses Manifest V2 with different APIs:
 * - browser_action instead of action
 * - sidebar_action instead of side_panel
 * - background scripts instead of service_worker
 */
export default defineConfig({
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  define: {
    // Define browser environment for conditional code
    'import.meta.env.BROWSER': JSON.stringify('firefox'),
  },
  build: {
    outDir: 'dist-firefox',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        'background/index': resolve(__dirname, 'src/background/index.ts'),
        'content/capture': resolve(__dirname, 'src/content/capture.ts'),
        'popup/index': resolve(__dirname, 'src/popup/index.html'),
        'sidepanel/index': resolve(__dirname, 'src/sidepanel/index.html'),
        'options/index': resolve(__dirname, 'src/options/index.html'),
      },
      output: {
        entryFileNames: (chunkInfo) => {
          // Keep the directory structure for entry points
          const name = chunkInfo.name || 'index';
          if (name.includes('/')) {
            return `${name}.js`;
          }
          return `[name].js`;
        },
        chunkFileNames: 'chunks/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
  plugins: [
    {
      name: 'copy-firefox-manifest',
      closeBundle() {
        // Copy Firefox manifest
        const distDir = resolve(__dirname, 'dist-firefox');
        if (!existsSync(distDir)) {
          mkdirSync(distDir, { recursive: true });
        }
        copyFileSync(
          resolve(__dirname, 'manifest.firefox.json'),
          resolve(distDir, 'manifest.json')
        );

        // Copy assets
        const assetsDir = resolve(distDir, 'assets');
        if (!existsSync(assetsDir)) {
          mkdirSync(assetsDir, { recursive: true });
        }

        // Copy icon files if they exist
        const iconsDir = resolve(__dirname, 'assets/icons');
        const distIconsDir = resolve(assetsDir, 'icons');
        if (existsSync(iconsDir)) {
          if (!existsSync(distIconsDir)) {
            mkdirSync(distIconsDir, { recursive: true });
          }
          const iconSizes = ['16', '32', '48', '128'];
          for (const size of iconSizes) {
            const srcIcon = resolve(iconsDir, `icon-${size}.png`);
            if (existsSync(srcIcon)) {
              copyFileSync(srcIcon, resolve(distIconsDir, `icon-${size}.png`));
            }
          }
        }

        console.log('Firefox extension built successfully!');
        console.log('Output: dist-firefox/');
        console.log('To test: Load dist-firefox/ in Firefox about:debugging');
      },
    },
  ],
});
