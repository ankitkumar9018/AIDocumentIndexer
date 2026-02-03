// AIDocumentIndexer Desktop - License Validation Module
// ======================================================
//
// Handles license validation for the desktop application.
// Supports multiple license providers (Keygen, Cryptolens, self-hosted).

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use log::{error, info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

// =============================================================================
// Types
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LicenseProvider {
    Keygen,
    Cryptolens,
    SelfHosted,
    Offline,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LicenseTier {
    Community,
    Professional,
    Team,
    Enterprise,
    Unlimited,
}

impl Default for LicenseTier {
    fn default() -> Self {
        LicenseTier::Community
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineFingerprint {
    pub machine_id: String,
    pub hostname: String,
    pub platform: String,
    pub architecture: String,
    pub fingerprint_hash: String,
}

impl MachineFingerprint {
    /// Generate fingerprint for the current machine
    pub fn generate() -> Self {
        let machine_id = Self::get_machine_id();
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());
        let platform = std::env::consts::OS.to_string();
        let architecture = std::env::consts::ARCH.to_string();

        // Create composite fingerprint hash
        let components = format!(
            "{}|{}|{}|{}",
            machine_id, hostname, platform, architecture
        );
        let mut hasher = Sha256::new();
        hasher.update(components.as_bytes());
        let fingerprint_hash = format!("{:x}", hasher.finalize())[..32].to_string();

        Self {
            machine_id,
            hostname,
            platform,
            architecture,
            fingerprint_hash,
        }
    }

    #[cfg(target_os = "macos")]
    fn get_machine_id() -> String {
        use std::process::Command;

        // Try to get IOPlatformSerialNumber
        if let Ok(output) = Command::new("ioreg")
            .args(["-rd1", "-c", "IOPlatformExpertDevice"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("IOPlatformSerialNumber") {
                    if let Some(serial) = line.split('"').nth(3) {
                        return serial.to_string();
                    }
                }
            }
        }

        Self::fallback_machine_id()
    }

    #[cfg(target_os = "linux")]
    fn get_machine_id() -> String {
        // Try /etc/machine-id first
        if let Ok(id) = std::fs::read_to_string("/etc/machine-id") {
            return id.trim().to_string();
        }

        // Fallback to /var/lib/dbus/machine-id
        if let Ok(id) = std::fs::read_to_string("/var/lib/dbus/machine-id") {
            return id.trim().to_string();
        }

        Self::fallback_machine_id()
    }

    #[cfg(target_os = "windows")]
    fn get_machine_id() -> String {
        use winreg::enums::*;
        use winreg::RegKey;

        if let Ok(hklm) = RegKey::predef(HKEY_LOCAL_MACHINE)
            .open_subkey("SOFTWARE\\Microsoft\\Cryptography")
        {
            if let Ok(guid) = hklm.get_value::<String, _>("MachineGuid") {
                return guid;
            }
        }

        Self::fallback_machine_id()
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    fn get_machine_id() -> String {
        Self::fallback_machine_id()
    }

    fn fallback_machine_id() -> String {
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let mut hasher = Sha256::new();
        hasher.update(hostname.as_bytes());
        format!("{:x}", hasher.finalize())[..32].to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    pub license_key: String,
    pub tier: LicenseTier,
    pub valid: bool,
    pub expires_at: Option<DateTime<Utc>>,
    pub activated_at: Option<DateTime<Utc>>,
    pub machine_fingerprint: Option<String>,
    pub max_users: Option<u32>,
    pub max_documents: Option<u32>,
    pub features: HashSet<String>,
    pub customer_name: Option<String>,
    pub error: Option<String>,
    pub last_validated: Option<DateTime<Utc>>,
    pub grace_period_ends: Option<DateTime<Utc>>,
}

impl Default for LicenseInfo {
    fn default() -> Self {
        Self {
            license_key: String::new(),
            tier: LicenseTier::Community,
            valid: false,
            expires_at: None,
            activated_at: None,
            machine_fingerprint: None,
            max_users: Some(1),
            max_documents: Some(100),
            features: HashSet::new(),
            customer_name: None,
            error: Some("No license".to_string()),
            last_validated: None,
            grace_period_ends: None,
        }
    }
}

impl LicenseInfo {
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Utc::now() > exp)
            .unwrap_or(false)
    }

    pub fn is_in_grace_period(&self) -> bool {
        self.grace_period_ends
            .map(|gp| Utc::now() < gp)
            .unwrap_or(false)
    }

    pub fn has_feature(&self, feature: &str) -> bool {
        if self.tier == LicenseTier::Unlimited {
            return true;
        }
        self.features.contains(feature) || self.get_tier_features().contains(feature)
    }

    fn get_tier_features(&self) -> HashSet<String> {
        let mut features = HashSet::new();

        match self.tier {
            LicenseTier::Community => {
                features.insert("basic_search".to_string());
                features.insert("document_upload".to_string());
                features.insert("basic_chat".to_string());
            }
            LicenseTier::Professional => {
                features.insert("basic_search".to_string());
                features.insert("advanced_search".to_string());
                features.insert("document_upload".to_string());
                features.insert("basic_chat".to_string());
                features.insert("advanced_chat".to_string());
                features.insert("knowledge_graph".to_string());
                features.insert("connectors".to_string());
            }
            LicenseTier::Team => {
                features.extend(LicenseInfo {
                    tier: LicenseTier::Professional,
                    ..Default::default()
                }.get_tier_features());
                features.insert("collaboration".to_string());
                features.insert("workflows".to_string());
            }
            LicenseTier::Enterprise | LicenseTier::Unlimited => {
                features.extend(LicenseInfo {
                    tier: LicenseTier::Team,
                    ..Default::default()
                }.get_tier_features());
                features.insert("sso".to_string());
                features.insert("audit_logs".to_string());
                features.insert("api_access".to_string());
                features.insert("custom_models".to_string());
            }
        }

        features
    }
}

// =============================================================================
// License Service
// =============================================================================

pub struct LicenseService {
    provider: LicenseProvider,
    server_url: String,
    api_key: String,
    product_id: String,
    http_client: Client,
    fingerprint: MachineFingerprint,
    cached_license: RwLock<Option<LicenseInfo>>,
    cache_ttl: Duration,
    grace_period_hours: u64,
}

impl LicenseService {
    pub fn new(
        provider: LicenseProvider,
        server_url: &str,
        api_key: &str,
    ) -> Self {
        Self {
            provider,
            server_url: server_url.to_string(),
            api_key: api_key.to_string(),
            product_id: "aidocindexer".to_string(),
            http_client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            fingerprint: MachineFingerprint::generate(),
            cached_license: RwLock::new(None),
            cache_ttl: Duration::from_secs(3600), // 1 hour
            grace_period_hours: 72, // 3 days
        }
    }

    /// Get the machine fingerprint
    pub fn get_fingerprint(&self) -> &MachineFingerprint {
        &self.fingerprint
    }

    /// Validate a license key
    pub async fn validate_license(&self, license_key: &str) -> Result<LicenseInfo> {
        // Check cache first
        if let Some(cached) = self.cached_license.read().await.as_ref() {
            if cached.license_key == license_key && cached.valid && !cached.is_expired() {
                return Ok(cached.clone());
            }
        }

        let result = match self.provider {
            LicenseProvider::Keygen => self.validate_keygen(license_key).await,
            LicenseProvider::Cryptolens => self.validate_cryptolens(license_key).await,
            LicenseProvider::SelfHosted => self.validate_self_hosted(license_key).await,
            LicenseProvider::Offline => self.validate_offline(license_key).await,
        };

        match result {
            Ok(license) => {
                *self.cached_license.write().await = Some(license.clone());
                Ok(license)
            }
            Err(e) => {
                // Check if we have a valid cached license in grace period
                if let Some(cached) = self.cached_license.read().await.as_ref() {
                    if cached.is_in_grace_period() {
                        warn!("Using cached license during validation failure");
                        return Ok(cached.clone());
                    }
                }
                Err(e)
            }
        }
    }

    /// Validate with Keygen.sh
    async fn validate_keygen(&self, license_key: &str) -> Result<LicenseInfo> {
        let account_id = std::env::var("KEYGEN_ACCOUNT_ID")
            .unwrap_or_else(|_| "".to_string());

        let response = self.http_client
            .post(&format!(
                "https://api.keygen.sh/v1/accounts/{}/licenses/actions/validate-key",
                account_id
            ))
            .header("Authorization", format!("License {}", license_key))
            .header("Content-Type", "application/vnd.api+json")
            .header("Accept", "application/vnd.api+json")
            .json(&serde_json::json!({
                "meta": {
                    "key": license_key,
                    "scope": {
                        "fingerprint": self.fingerprint.fingerprint_hash
                    }
                }
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("License validation failed"));
        }

        let data: serde_json::Value = response.json().await?;

        let valid = data["meta"]["valid"].as_bool().unwrap_or(false);
        if !valid {
            return Ok(LicenseInfo {
                license_key: license_key.to_string(),
                tier: LicenseTier::Community,
                valid: false,
                error: Some(data["meta"]["detail"].as_str().unwrap_or("Invalid license").to_string()),
                ..Default::default()
            });
        }

        let attrs = &data["data"]["attributes"];
        let metadata = &attrs["metadata"];

        Ok(LicenseInfo {
            license_key: license_key.to_string(),
            tier: metadata["tier"].as_str()
                .and_then(|t| serde_json::from_str(&format!("\"{}\"", t)).ok())
                .unwrap_or(LicenseTier::Professional),
            valid: true,
            expires_at: attrs["expiry"].as_str()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            activated_at: Some(Utc::now()),
            machine_fingerprint: Some(self.fingerprint.fingerprint_hash.clone()),
            features: metadata["features"].as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            max_users: metadata["max_users"].as_u64().map(|n| n as u32),
            max_documents: metadata["max_documents"].as_u64().map(|n| n as u32),
            customer_name: attrs["name"].as_str().map(String::from),
            error: None,
            last_validated: Some(Utc::now()),
            grace_period_ends: Some(Utc::now() + chrono::Duration::hours(self.grace_period_hours as i64)),
        })
    }

    /// Validate with Cryptolens
    async fn validate_cryptolens(&self, license_key: &str) -> Result<LicenseInfo> {
        let response = self.http_client
            .post("https://app.cryptolens.io/api/key/Activate")
            .json(&serde_json::json!({
                "token": self.api_key,
                "ProductId": self.product_id,
                "Key": license_key,
                "MachineCode": self.fingerprint.fingerprint_hash
            }))
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;

        if data["result"].as_i64() != Some(0) {
            return Ok(LicenseInfo {
                license_key: license_key.to_string(),
                tier: LicenseTier::Community,
                valid: false,
                error: Some(data["message"].as_str().unwrap_or("Validation failed").to_string()),
                ..Default::default()
            });
        }

        let license_data = &data["licenseKey"];
        let expires = license_data["expires"].as_str()
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        // Parse features from data objects
        let mut features = HashSet::new();
        if let Some(data_objects) = license_data["dataObjects"].as_array() {
            for obj in data_objects {
                if obj["name"].as_str() == Some("features") {
                    if let Some(value) = obj["stringValue"].as_str() {
                        features = value.split(',').map(String::from).collect();
                    }
                }
            }
        }

        Ok(LicenseInfo {
            license_key: license_key.to_string(),
            tier: Self::determine_tier_from_features(&features),
            valid: true,
            expires_at: expires,
            activated_at: Some(Utc::now()),
            machine_fingerprint: Some(self.fingerprint.fingerprint_hash.clone()),
            features,
            customer_name: license_data["notes"].as_str().map(String::from),
            error: None,
            last_validated: Some(Utc::now()),
            grace_period_ends: Some(Utc::now() + chrono::Duration::hours(self.grace_period_hours as i64)),
            ..Default::default()
        })
    }

    /// Validate with self-hosted license server
    async fn validate_self_hosted(&self, license_key: &str) -> Result<LicenseInfo> {
        let response = self.http_client
            .post(&format!("{}/api/v1/licenses/validate", self.server_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "license_key": license_key,
                "machine_fingerprint": self.fingerprint.fingerprint_hash,
                "product_id": self.product_id,
                "hostname": self.fingerprint.hostname,
                "platform": self.fingerprint.platform
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("License server returned error"));
        }

        let data: serde_json::Value = response.json().await?;

        if !data["valid"].as_bool().unwrap_or(false) {
            return Ok(LicenseInfo {
                license_key: license_key.to_string(),
                tier: LicenseTier::Community,
                valid: false,
                error: Some(data["error"].as_str().unwrap_or("Validation failed").to_string()),
                ..Default::default()
            });
        }

        Ok(LicenseInfo {
            license_key: license_key.to_string(),
            tier: data["tier"].as_str()
                .and_then(|t| serde_json::from_str(&format!("\"{}\"", t)).ok())
                .unwrap_or(LicenseTier::Professional),
            valid: true,
            expires_at: data["expires_at"].as_str()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            activated_at: Some(Utc::now()),
            machine_fingerprint: Some(self.fingerprint.fingerprint_hash.clone()),
            features: data["features"].as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default(),
            max_users: data["max_users"].as_u64().map(|n| n as u32),
            max_documents: data["max_documents"].as_u64().map(|n| n as u32),
            customer_name: data["customer_name"].as_str().map(String::from),
            error: None,
            last_validated: Some(Utc::now()),
            grace_period_ends: Some(Utc::now() + chrono::Duration::hours(self.grace_period_hours as i64)),
        })
    }

    /// Validate offline license from file
    async fn validate_offline(&self, license_key: &str) -> Result<LicenseInfo> {
        let license_path = dirs::home_dir()
            .ok_or_else(|| anyhow!("Cannot find home directory"))?
            .join(".aidocindexer")
            .join("license.key");

        if !license_path.exists() {
            return Ok(LicenseInfo {
                license_key: license_key.to_string(),
                tier: LicenseTier::Community,
                valid: false,
                error: Some("License file not found".to_string()),
                ..Default::default()
            });
        }

        let content = tokio::fs::read_to_string(&license_path).await?;
        let mut data = std::collections::HashMap::new();
        let mut signature = String::new();

        for line in content.lines() {
            if line.starts_with("SIGNATURE=") {
                signature = line[10..].to_string();
            } else if let Some((key, value)) = line.split_once('=') {
                data.insert(key.to_string(), value.to_string());
            }
        }

        // Verify signature
        let signing_key = std::env::var("LICENSE_SIGNING_KEY").unwrap_or_default();
        if signing_key.is_empty() {
            return Ok(LicenseInfo {
                license_key: license_key.to_string(),
                tier: LicenseTier::Community,
                valid: false,
                error: Some("License signing key not configured".to_string()),
                ..Default::default()
            });
        }

        // Compute expected signature (HMAC-SHA256)
        use hmac::{Hmac, Mac};
        type HmacSha256 = Hmac<Sha256>;

        let mut sorted_data: Vec<_> = data.iter().collect();
        sorted_data.sort_by_key(|(k, _)| k.as_str());
        let content_to_sign: String = sorted_data
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("\n");

        let mut mac = HmacSha256::new_from_slice(signing_key.as_bytes())
            .map_err(|_| anyhow!("Invalid signing key"))?;
        mac.update(content_to_sign.as_bytes());
        let expected_sig = format!("{:x}", mac.finalize().into_bytes());

        if signature != expected_sig {
            return Ok(LicenseInfo {
                license_key: license_key.to_string(),
                tier: LicenseTier::Community,
                valid: false,
                error: Some("Invalid license signature".to_string()),
                ..Default::default()
            });
        }

        // Verify fingerprint matches
        if let Some(fp) = data.get("FINGERPRINT") {
            if fp != &self.fingerprint.fingerprint_hash {
                return Ok(LicenseInfo {
                    license_key: license_key.to_string(),
                    tier: LicenseTier::Community,
                    valid: false,
                    error: Some("License not valid for this machine".to_string()),
                    ..Default::default()
                });
            }
        }

        // Parse features
        let features: HashSet<String> = data.get("FEATURES")
            .map(|f| f.split(',').map(String::from).collect())
            .unwrap_or_default();

        Ok(LicenseInfo {
            license_key: data.get("LICENSE_KEY").cloned().unwrap_or_else(|| license_key.to_string()),
            tier: data.get("TIER")
                .and_then(|t| serde_json::from_str(&format!("\"{}\"", t.to_lowercase())).ok())
                .unwrap_or(LicenseTier::Professional),
            valid: true,
            expires_at: data.get("EXPIRES")
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            machine_fingerprint: Some(self.fingerprint.fingerprint_hash.clone()),
            features,
            max_users: data.get("MAX_USERS").and_then(|s| s.parse().ok()),
            max_documents: data.get("MAX_DOCUMENTS").and_then(|s| s.parse().ok()),
            customer_name: data.get("CUSTOMER").cloned(),
            error: None,
            last_validated: Some(Utc::now()),
            ..Default::default()
        })
    }

    fn determine_tier_from_features(features: &HashSet<String>) -> LicenseTier {
        if features.contains("unlimited") || features.contains("all") {
            LicenseTier::Unlimited
        } else if features.contains("sso") || features.contains("audit_logs") {
            LicenseTier::Enterprise
        } else if features.contains("collaboration") || features.contains("workflows") {
            LicenseTier::Team
        } else if features.contains("advanced_search") || features.contains("knowledge_graph") {
            LicenseTier::Professional
        } else {
            LicenseTier::Community
        }
    }

    /// Get cached license info
    pub async fn get_current_license(&self) -> Option<LicenseInfo> {
        self.cached_license.read().await.clone()
    }

    /// Check if a feature is available
    pub async fn has_feature(&self, feature: &str) -> bool {
        self.cached_license
            .read()
            .await
            .as_ref()
            .map(|l| l.has_feature(feature))
            .unwrap_or(false)
    }
}

// =============================================================================
// Tauri Commands
// =============================================================================

/// Get machine fingerprint
#[tauri::command]
pub fn get_machine_fingerprint() -> MachineFingerprint {
    MachineFingerprint::generate()
}

/// Validate license key
#[tauri::command]
pub async fn validate_license(
    license_key: String,
    provider: LicenseProvider,
    server_url: Option<String>,
    api_key: Option<String>,
) -> Result<LicenseInfo, String> {
    let service = LicenseService::new(
        provider,
        &server_url.unwrap_or_else(|| "https://license.example.com".to_string()),
        &api_key.unwrap_or_default(),
    );

    service
        .validate_license(&license_key)
        .await
        .map_err(|e| e.to_string())
}

/// Get license info (current cached license)
#[tauri::command]
pub async fn get_license_info() -> Option<LicenseInfo> {
    // In practice, this would read from app state
    None
}

/// Check if feature is available
#[tauri::command]
pub fn has_license_feature(feature: String) -> bool {
    // In practice, this would check app state
    // For now, return true in dev mode
    #[cfg(debug_assertions)]
    return true;

    #[cfg(not(debug_assertions))]
    return false;
}
