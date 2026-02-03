/**
 * AIDocumentIndexer Content Script
 * =================================
 *
 * Captures page content for indexing.
 * Extracts main content, handles dynamic pages, and converts to clean text.
 */

import { MessageType } from '../shared/types';

// Content extraction configuration
const EXCLUDED_TAGS = new Set([
  'SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME', 'OBJECT', 'EMBED',
  'SVG', 'CANVAS', 'VIDEO', 'AUDIO', 'NAV', 'FOOTER', 'HEADER',
  'ASIDE', 'FORM', 'INPUT', 'BUTTON', 'SELECT', 'TEXTAREA',
]);

const EXCLUDED_CLASSES = [
  'nav', 'navigation', 'menu', 'sidebar', 'footer', 'header',
  'ad', 'advertisement', 'popup', 'modal', 'cookie', 'banner',
  'social', 'share', 'comment', 'comments', 'related',
];

const EXCLUDED_IDS = [
  'nav', 'navigation', 'menu', 'sidebar', 'footer', 'header',
  'ad', 'advertisement', 'popup', 'modal', 'cookie', 'banner',
];

/**
 * Extract the main content from the page
 */
function extractMainContent(): string {
  // Try to find main content container
  const mainContent = document.querySelector('main, article, [role="main"], .main-content, #main-content, .post-content, .article-content');

  if (mainContent) {
    return cleanText(mainContent.textContent || '');
  }

  // Fallback: extract from body while excluding common non-content elements
  const body = document.body;
  if (!body) return '';

  // Clone body to avoid modifying the actual DOM
  const bodyClone = body.cloneNode(true) as HTMLElement;

  // Remove excluded elements
  removeExcludedElements(bodyClone);

  return cleanText(bodyClone.textContent || '');
}

/**
 * Remove elements that shouldn't be included in content
 */
function removeExcludedElements(container: HTMLElement): void {
  // Remove by tag name
  EXCLUDED_TAGS.forEach((tag) => {
    const elements = container.getElementsByTagName(tag);
    for (let i = elements.length - 1; i >= 0; i--) {
      elements[i].remove();
    }
  });

  // Remove by class
  EXCLUDED_CLASSES.forEach((className) => {
    const elements = container.querySelectorAll(`[class*="${className}"]`);
    elements.forEach((el) => el.remove());
  });

  // Remove by ID
  EXCLUDED_IDS.forEach((id) => {
    const elements = container.querySelectorAll(`[id*="${id}"]`);
    elements.forEach((el) => el.remove());
  });

  // Remove elements with data-noindex attribute
  const noIndexElements = container.querySelectorAll('[data-noindex], [data-no-index]');
  noIndexElements.forEach((el) => el.remove());
}

/**
 * Clean and normalize extracted text
 */
function cleanText(text: string): string {
  return text
    // Replace multiple whitespace with single space
    .replace(/\s+/g, ' ')
    // Remove leading/trailing whitespace from each line
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .join('\n')
    // Final trim
    .trim();
}

/**
 * Extract metadata from the page
 */
function extractMetadata(): Record<string, string> {
  const metadata: Record<string, string> = {
    url: window.location.href,
    title: document.title,
  };

  // Open Graph metadata
  const ogTitle = document.querySelector('meta[property="og:title"]');
  if (ogTitle) metadata.ogTitle = ogTitle.getAttribute('content') || '';

  const ogDescription = document.querySelector('meta[property="og:description"]');
  if (ogDescription) metadata.description = ogDescription.getAttribute('content') || '';

  const ogImage = document.querySelector('meta[property="og:image"]');
  if (ogImage) metadata.image = ogImage.getAttribute('content') || '';

  // Standard metadata
  const description = document.querySelector('meta[name="description"]');
  if (description && !metadata.description) {
    metadata.description = description.getAttribute('content') || '';
  }

  const author = document.querySelector('meta[name="author"]');
  if (author) metadata.author = author.getAttribute('content') || '';

  const keywords = document.querySelector('meta[name="keywords"]');
  if (keywords) metadata.keywords = keywords.getAttribute('content') || '';

  // Article-specific metadata
  const publishedTime = document.querySelector('meta[property="article:published_time"]');
  if (publishedTime) metadata.publishedTime = publishedTime.getAttribute('content') || '';

  // Canonical URL
  const canonical = document.querySelector('link[rel="canonical"]');
  if (canonical) metadata.canonicalUrl = canonical.getAttribute('href') || '';

  return metadata;
}

/**
 * Get the page type (article, product, search, etc.)
 */
function getPageType(): string {
  const schemaType = document.querySelector('[itemtype]')?.getAttribute('itemtype') || '';

  if (schemaType.includes('Article') || schemaType.includes('BlogPosting')) return 'article';
  if (schemaType.includes('Product')) return 'product';
  if (schemaType.includes('Recipe')) return 'recipe';

  // Heuristic detection
  if (document.querySelector('article')) return 'article';
  if (document.querySelector('.product, [data-product]')) return 'product';
  if (window.location.pathname.includes('/search')) return 'search';

  return 'webpage';
}

/**
 * Capture full page data
 */
function capturePageData(): {
  url: string;
  title: string;
  content: string;
  metadata: Record<string, string>;
  pageType: string;
  capturedAt: string;
} {
  return {
    url: window.location.href,
    title: document.title,
    content: extractMainContent(),
    metadata: extractMetadata(),
    pageType: getPageType(),
    capturedAt: new Date().toISOString(),
  };
}

/**
 * Get selected text if any
 */
function getSelectedText(): string | null {
  const selection = window.getSelection();
  if (selection && selection.toString().trim()) {
    return selection.toString().trim();
  }
  return null;
}

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === MessageType.CAPTURE_PAGE) {
    try {
      const pageData = capturePageData();
      sendResponse({ success: true, data: pageData });
    } catch (error) {
      sendResponse({ success: false, error: (error as Error).message });
    }
    return true; // Will respond asynchronously
  }

  if (message.type === 'GET_SELECTION') {
    const selectedText = getSelectedText();
    sendResponse({ success: true, data: selectedText });
    return true;
  }

  if (message.type === 'GET_PAGE_INFO') {
    sendResponse({
      success: true,
      data: {
        url: window.location.href,
        title: document.title,
        hasSelection: !!getSelectedText(),
      },
    });
    return true;
  }
});

// Export for testing
export {
  extractMainContent,
  extractMetadata,
  getPageType,
  capturePageData,
  cleanText,
};
