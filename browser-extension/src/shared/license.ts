/**
 * AIDocumentIndexer Browser Extension - License Utilities
 * ========================================================
 *
 * Handles license validation and feature gating for the browser extension.
 * Since Chrome Web Store prohibits code obfuscation, the extension uses
 * server-side license validation via API calls.
 */

import { storage } from './browser-polyfill';

// =============================================================================
// Types
// =============================================================================

export type LicenseTier =
  | 'community'
  | 'professional'
  | 'team'
  | 'enterprise'
  | 'unlimited';

export interface LicenseInfo {
  valid: boolean;
  tier: LicenseTier;
  tierDisplay: string;
  expiresAt: string | null;
  daysUntilExpiry: number | null;
  features: string[];
  maxDocuments: number | null;
  customerName: string | null;
  error: string | null;
  lastChecked: string;
}

export interface LicenseSettings {
  serverUrl: string;
  apiKey: string;
  licenseKey?: string;
}

// Default license info (community tier)
const DEFAULT_LICENSE: LicenseInfo = {
  valid: false,
  tier: 'community',
  tierDisplay: 'Community (Free)',
  expiresAt: null,
  daysUntilExpiry: null,
  features: ['basic_search', 'document_upload', 'basic_chat'],
  maxDocuments: 100,
  customerName: null,
  error: 'License not checked',
  lastChecked: new Date().toISOString(),
};

// Feature flags by tier
const TIER_FEATURES: Record<LicenseTier, string[]> = {
  community: ['basic_search', 'document_upload', 'basic_chat'],
  professional: [
    'basic_search',
    'advanced_search',
    'document_upload',
    'basic_chat',
    'advanced_chat',
    'knowledge_graph',
    'connectors',
  ],
  team: [
    'basic_search',
    'advanced_search',
    'document_upload',
    'basic_chat',
    'advanced_chat',
    'knowledge_graph',
    'connectors',
    'collaboration',
    'workflows',
  ],
  enterprise: [
    'basic_search',
    'advanced_search',
    'document_upload',
    'basic_chat',
    'advanced_chat',
    'knowledge_graph',
    'connectors',
    'collaboration',
    'workflows',
    'sso',
    'audit_logs',
    'api_access',
    'custom_models',
  ],
  unlimited: [], // All features enabled
};

// Cache TTL in milliseconds (1 hour)
const CACHE_TTL = 60 * 60 * 1000;

// =============================================================================
// Storage Keys
// =============================================================================

const STORAGE_KEYS = {
  LICENSE_INFO: 'aidoc_license_info',
  LICENSE_CHECKED_AT: 'aidoc_license_checked_at',
  SETTINGS: 'aidoc_settings',
};

// =============================================================================
// License Service
// =============================================================================

/**
 * Get stored license info from extension storage
 */
export async function getCachedLicense(): Promise<LicenseInfo | null> {
  const data = await storage.get<{
    [STORAGE_KEYS.LICENSE_INFO]: LicenseInfo;
    [STORAGE_KEYS.LICENSE_CHECKED_AT]: string;
  }>([STORAGE_KEYS.LICENSE_INFO, STORAGE_KEYS.LICENSE_CHECKED_AT]);

  const licenseInfo = data[STORAGE_KEYS.LICENSE_INFO];
  const checkedAt = data[STORAGE_KEYS.LICENSE_CHECKED_AT];

  if (!licenseInfo || !checkedAt) {
    return null;
  }

  // Check if cache is still valid
  const checkedTime = new Date(checkedAt).getTime();
  if (Date.now() - checkedTime > CACHE_TTL) {
    return null; // Cache expired
  }

  return licenseInfo;
}

/**
 * Store license info in extension storage
 */
export async function cacheLicense(licenseInfo: LicenseInfo): Promise<void> {
  await storage.set({
    [STORAGE_KEYS.LICENSE_INFO]: licenseInfo,
    [STORAGE_KEYS.LICENSE_CHECKED_AT]: new Date().toISOString(),
  });
}

/**
 * Get extension settings
 */
export async function getSettings(): Promise<LicenseSettings> {
  const data = await storage.get<{
    [STORAGE_KEYS.SETTINGS]: LicenseSettings;
  }>([STORAGE_KEYS.SETTINGS]);

  return (
    data[STORAGE_KEYS.SETTINGS] || {
      serverUrl: 'http://localhost:8000',
      apiKey: '',
    }
  );
}

/**
 * Validate license with the server
 */
export async function validateLicense(
  settings?: LicenseSettings,
  forceRefresh = false
): Promise<LicenseInfo> {
  // Check cache first (unless force refresh)
  if (!forceRefresh) {
    const cached = await getCachedLicense();
    if (cached) {
      return cached;
    }
  }

  // Get settings if not provided
  const config = settings || (await getSettings());

  if (!config.serverUrl) {
    return {
      ...DEFAULT_LICENSE,
      error: 'Server URL not configured',
    };
  }

  try {
    const response = await fetch(`${config.serverUrl}/api/v1/license/info`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
      },
    });

    if (!response.ok) {
      // Server returned error, use default community license
      const error = await response.text();
      const licenseInfo: LicenseInfo = {
        ...DEFAULT_LICENSE,
        error: `Server error: ${response.status}`,
        lastChecked: new Date().toISOString(),
      };
      await cacheLicense(licenseInfo);
      return licenseInfo;
    }

    const data = await response.json();

    const licenseInfo: LicenseInfo = {
      valid: data.valid,
      tier: data.tier as LicenseTier,
      tierDisplay: data.tier_display || data.tier,
      expiresAt: data.expires_at,
      daysUntilExpiry: data.days_until_expiry,
      features: data.features || TIER_FEATURES[data.tier as LicenseTier] || [],
      maxDocuments: data.max_documents,
      customerName: data.customer_name,
      error: data.error || null,
      lastChecked: new Date().toISOString(),
    };

    // Cache the result
    await cacheLicense(licenseInfo);

    return licenseInfo;
  } catch (error) {
    console.error('License validation failed:', error);

    // Check if we have a cached license to use during network failure
    const cached = await getCachedLicense();
    if (cached) {
      return cached;
    }

    return {
      ...DEFAULT_LICENSE,
      error: `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
      lastChecked: new Date().toISOString(),
    };
  }
}

/**
 * Check if a specific feature is available
 */
export async function hasFeature(feature: string): Promise<boolean> {
  const license = await validateLicense();

  if (!license.valid) {
    // Even without a valid license, allow community features
    return TIER_FEATURES.community.includes(feature);
  }

  if (license.tier === 'unlimited') {
    return true;
  }

  // Check if feature is in the license features or tier features
  if (license.features.includes(feature)) {
    return true;
  }

  const tierFeatures = TIER_FEATURES[license.tier] || [];
  return tierFeatures.includes(feature);
}

/**
 * Get current license tier
 */
export async function getCurrentTier(): Promise<LicenseTier> {
  const license = await validateLicense();
  return license.tier;
}

/**
 * Check if the license allows more documents
 */
export async function canAddDocument(currentCount: number): Promise<boolean> {
  const license = await validateLicense();

  if (license.tier === 'unlimited' || !license.maxDocuments) {
    return true;
  }

  return currentCount < license.maxDocuments;
}

// =============================================================================
// Feature Gating Helpers
// =============================================================================

/**
 * Show upgrade prompt for a feature
 */
export function showUpgradePrompt(feature: string): void {
  // In the browser extension, we show a notification or redirect to upgrade page
  const featureNames: Record<string, string> = {
    advanced_search: 'Advanced Search',
    knowledge_graph: 'Knowledge Graph',
    connectors: 'Data Connectors',
    collaboration: 'Team Collaboration',
    workflows: 'Automated Workflows',
    sso: 'Single Sign-On',
    audit_logs: 'Audit Logging',
    custom_models: 'Custom AI Models',
  };

  const featureName = featureNames[feature] || feature;

  console.log(
    `Feature "${featureName}" requires a higher license tier. Please upgrade to unlock this feature.`
  );
}

/**
 * Decorator-style function for gating features
 * Usage: await gateFeature('knowledge_graph', async () => { ... })
 */
export async function gateFeature<T>(
  feature: string,
  action: () => Promise<T>
): Promise<T | null> {
  const available = await hasFeature(feature);

  if (!available) {
    showUpgradePrompt(feature);
    return null;
  }

  return action();
}

// =============================================================================
// Tier Display Helpers
// =============================================================================

export const TIER_DISPLAY_NAMES: Record<LicenseTier, string> = {
  community: 'Community (Free)',
  professional: 'Professional',
  team: 'Team',
  enterprise: 'Enterprise',
  unlimited: 'Unlimited',
};

export const TIER_COLORS: Record<LicenseTier, string> = {
  community: '#6B7280', // Gray
  professional: '#3B82F6', // Blue
  team: '#10B981', // Green
  enterprise: '#8B5CF6', // Purple
  unlimited: '#F59E0B', // Amber
};

/**
 * Get display name for a tier
 */
export function getTierDisplayName(tier: LicenseTier): string {
  return TIER_DISPLAY_NAMES[tier] || tier;
}

/**
 * Get color for a tier (for UI)
 */
export function getTierColor(tier: LicenseTier): string {
  return TIER_COLORS[tier] || '#6B7280';
}
