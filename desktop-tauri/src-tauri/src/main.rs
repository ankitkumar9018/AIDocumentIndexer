// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Manager, State, SystemTray, SystemTrayEvent, SystemTrayMenu,
    CustomMenuItem, WindowEvent,
};
use tokio::sync::RwLock;

mod database;
mod embeddings;
mod file_watcher;
mod licensing;
mod ollama;
mod server_client;

use database::LocalDatabase;
use file_watcher::FileWatcher;
use ollama::OllamaClient;
use server_client::ServerClient;

// =============================================================================
// Application State
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AppMode {
    Local,  // Offline mode with Ollama
    Server, // Connected to server
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub mode: AppMode,
    pub server_url: String,
    pub api_key: Option<String>,
    pub ollama_url: String,
    pub ollama_model: String,
    pub embedding_model: String,
    pub watch_directories: Vec<PathBuf>,
    pub auto_index_on_change: bool,
    pub global_shortcut: String,
    pub start_minimized: bool,
    pub show_notifications: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            mode: AppMode::Local,
            server_url: "http://localhost:8000".to_string(),
            api_key: None,
            ollama_url: "http://localhost:11434".to_string(),
            ollama_model: "llama3.2".to_string(),
            embedding_model: "nomic-embed-text".to_string(),
            watch_directories: vec![],
            auto_index_on_change: true,
            global_shortcut: "CommandOrControl+Shift+Space".to_string(),
            start_minimized: false,
            show_notifications: true,
        }
    }
}

pub struct AppState {
    pub settings: RwLock<AppSettings>,
    pub database: Arc<LocalDatabase>,
    pub ollama: Arc<OllamaClient>,
    pub server: Arc<ServerClient>,
    pub file_watcher: RwLock<Option<FileWatcher>>,
}

impl AppState {
    pub async fn new() -> Result<Self> {
        let settings = AppSettings::default();
        let database = Arc::new(LocalDatabase::new()?);
        let ollama = Arc::new(OllamaClient::new(&settings.ollama_url));
        let server = Arc::new(ServerClient::new(&settings.server_url));

        Ok(Self {
            settings: RwLock::new(settings),
            database,
            ollama,
            server,
            file_watcher: RwLock::new(None),
        })
    }
}

// =============================================================================
// Tauri Commands
// =============================================================================

/// Get current app mode
#[tauri::command]
async fn get_mode(state: State<'_, Arc<AppState>>) -> Result<AppMode, String> {
    Ok(state.settings.read().await.mode)
}

/// Set app mode (Local or Server)
#[tauri::command]
async fn set_mode(
    mode: AppMode,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    settings.mode = mode;
    info!("Mode changed to: {:?}", mode);
    Ok(())
}

/// Get app settings
#[tauri::command]
async fn get_settings(state: State<'_, Arc<AppState>>) -> Result<AppSettings, String> {
    Ok(state.settings.read().await.clone())
}

/// Update app settings
#[tauri::command]
async fn update_settings(
    settings: AppSettings,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut current = state.settings.write().await;
    *current = settings;
    info!("Settings updated");
    Ok(())
}

/// Check Ollama connection
#[tauri::command]
async fn check_ollama(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    match state.ollama.health_check().await {
        Ok(_) => Ok(true),
        Err(e) => {
            warn!("Ollama not available: {}", e);
            Ok(false)
        }
    }
}

/// Check server connection
#[tauri::command]
async fn check_server(state: State<'_, Arc<AppState>>) -> Result<bool, String> {
    let settings = state.settings.read().await;
    match state.server.health_check(&settings.server_url).await {
        Ok(_) => Ok(true),
        Err(e) => {
            warn!("Server not available: {}", e);
            Ok(false)
        }
    }
}

/// Get available Ollama models
#[tauri::command]
async fn get_ollama_models(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<String>, String> {
    state
        .ollama
        .list_models()
        .await
        .map_err(|e| e.to_string())
}

/// Chat with documents (LOCAL MODE)
#[tauri::command]
async fn chat_local(
    message: String,
    state: State<'_, Arc<AppState>>,
) -> Result<ChatResponse, String> {
    let settings = state.settings.read().await;

    // Search for relevant chunks
    let chunks = state
        .database
        .search(&message, 5)
        .map_err(|e| e.to_string())?;

    // Build context from chunks
    let context = chunks
        .iter()
        .map(|c| c.content.as_str())
        .collect::<Vec<_>>()
        .join("\n\n---\n\n");

    // Create prompt with context
    let prompt = format!(
        "You are a helpful assistant. Answer the user's question based on the following context from their documents.\n\n\
        Context:\n{}\n\n\
        User Question: {}\n\n\
        Answer:",
        context, message
    );

    // Generate response with Ollama
    let response = state
        .ollama
        .generate(&settings.ollama_model, &prompt)
        .await
        .map_err(|e| e.to_string())?;

    Ok(ChatResponse {
        content: response,
        sources: chunks.into_iter().map(|c| c.into()).collect(),
    })
}

/// Chat with documents (SERVER MODE)
#[tauri::command]
async fn chat_server(
    message: String,
    collection_id: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<ChatResponse, String> {
    let settings = state.settings.read().await;

    state
        .server
        .chat(&settings.server_url, &settings.api_key, &message, collection_id)
        .await
        .map_err(|e| e.to_string())
}

/// Search documents (LOCAL MODE)
#[tauri::command]
async fn search_local(
    query: String,
    top_k: Option<usize>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<SearchResult>, String> {
    let results = state
        .database
        .search(&query, top_k.unwrap_or(10))
        .map_err(|e| e.to_string())?;

    Ok(results.into_iter().map(|r| r.into()).collect())
}

/// Search documents (SERVER MODE)
#[tauri::command]
async fn search_server(
    query: String,
    collection_id: Option<String>,
    top_k: Option<usize>,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<SearchResult>, String> {
    let settings = state.settings.read().await;

    state
        .server
        .search(
            &settings.server_url,
            &settings.api_key,
            &query,
            collection_id,
            top_k,
        )
        .await
        .map_err(|e| e.to_string())
}

/// Index a file (LOCAL MODE)
#[tauri::command]
async fn index_file(
    path: PathBuf,
    state: State<'_, Arc<AppState>>,
) -> Result<IndexResult, String> {
    let settings = state.settings.read().await;

    // Read file content
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;

    // Extract file info
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Chunk the content
    let chunks = chunk_text(&content, 512, 50);

    // Generate embeddings with Ollama
    let embeddings = state
        .ollama
        .embed(&settings.embedding_model, &chunks)
        .await
        .map_err(|e| e.to_string())?;

    // Store in database
    let doc_id = state
        .database
        .insert_document(&file_name, path.to_string_lossy().as_ref())
        .map_err(|e| e.to_string())?;

    for (i, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
        state
            .database
            .insert_chunk(doc_id, i as i32, chunk, embedding)
            .map_err(|e| e.to_string())?;
    }

    info!("Indexed file: {} ({} chunks)", file_name, chunks.len());

    Ok(IndexResult {
        document_id: doc_id.to_string(),
        chunks_created: chunks.len(),
    })
}

/// Upload file to server (SERVER MODE)
#[tauri::command]
async fn upload_file(
    path: PathBuf,
    collection_id: Option<String>,
    state: State<'_, Arc<AppState>>,
) -> Result<IndexResult, String> {
    let settings = state.settings.read().await;

    state
        .server
        .upload_file(&settings.server_url, &settings.api_key, path, collection_id)
        .await
        .map_err(|e| e.to_string())
}

/// Get indexed documents (LOCAL MODE)
#[tauri::command]
async fn get_documents(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<DocumentInfo>, String> {
    state
        .database
        .list_documents()
        .map_err(|e| e.to_string())
}

/// Delete document (LOCAL MODE)
#[tauri::command]
async fn delete_document(
    document_id: String,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let id: i64 = document_id.parse().map_err(|_| "Invalid document ID")?;
    state.database.delete_document(id).map_err(|e| e.to_string())
}

/// Add watch directory
#[tauri::command]
async fn add_watch_directory(
    path: PathBuf,
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    if !settings.watch_directories.contains(&path) {
        settings.watch_directories.push(path.clone());
    }

    // Restart file watcher
    drop(settings);
    restart_file_watcher(state.inner().clone(), app).await;

    Ok(())
}

/// Remove watch directory
#[tauri::command]
async fn remove_watch_directory(
    path: PathBuf,
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<(), String> {
    let mut settings = state.settings.write().await;
    settings.watch_directories.retain(|p| p != &path);

    // Restart file watcher
    drop(settings);
    restart_file_watcher(state.inner().clone(), app).await;

    Ok(())
}

/// Get database stats
#[tauri::command]
async fn get_stats(state: State<'_, Arc<AppState>>) -> Result<DatabaseStats, String> {
    state.database.get_stats().map_err(|e| e.to_string())
}

// =============================================================================
// Response Types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub content: String,
    pub sources: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub document_id: String,
    pub document_name: String,
    pub content: String,
    pub score: f32,
    pub chunk_index: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexResult {
    pub document_id: String,
    pub chunks_created: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub chunk_count: i64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStats {
    pub document_count: i64,
    pub chunk_count: i64,
    pub total_size_bytes: i64,
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Simple text chunking
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();

    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);

        if end >= words.len() {
            break;
        }
        start += chunk_size - overlap;
    }

    chunks
}

/// Restart file watcher with current settings
async fn restart_file_watcher(state: Arc<AppState>, app: AppHandle) {
    let settings = state.settings.read().await;
    let directories = settings.watch_directories.clone();
    let auto_index = settings.auto_index_on_change;
    drop(settings);

    let mut watcher_guard = state.file_watcher.write().await;

    // Stop existing watcher
    if let Some(mut watcher) = watcher_guard.take() {
        watcher.stop();
    }

    if !directories.is_empty() && auto_index {
        let state_clone = state.clone();
        let app_clone = app.clone();

        let watcher = FileWatcher::new(directories, move |path| {
            let state = state_clone.clone();
            let app = app_clone.clone();

            tokio::spawn(async move {
                if let Err(e) = handle_file_change(state, app, path).await {
                    error!("Failed to handle file change: {}", e);
                }
            });
        });

        *watcher_guard = Some(watcher);
        info!("File watcher started");
    }
}

/// Handle file change event
async fn handle_file_change(
    state: Arc<AppState>,
    app: AppHandle,
    path: PathBuf,
) -> Result<()> {
    info!("File changed: {:?}", path);

    // Check if it's a supported file type
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let supported = matches!(
        extension.to_lowercase().as_str(),
        "txt" | "md" | "pdf" | "docx" | "doc" | "rtf" | "html" | "json" | "csv"
    );

    if !supported {
        return Ok(());
    }

    // Index the file
    let settings = state.settings.read().await;
    let content = std::fs::read_to_string(&path)?;
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

    let chunks = chunk_text(&content, 512, 50);
    let embeddings = state.ollama.embed(&settings.embedding_model, &chunks).await?;

    // Check if document exists
    let existing = state.database.find_document_by_path(path.to_string_lossy().as_ref())?;

    let doc_id = if let Some(doc) = existing {
        // Delete old chunks and update
        state.database.delete_chunks_for_document(doc.id)?;
        doc.id
    } else {
        // Insert new document
        state.database.insert_document(file_name, path.to_string_lossy().as_ref())?
    };

    for (i, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
        state.database.insert_chunk(doc_id, i as i32, chunk, embedding)?;
    }

    // Send notification to frontend
    app.emit_all("file-indexed", serde_json::json!({
        "path": path.to_string_lossy(),
        "document_id": doc_id,
        "chunks": chunks.len(),
    }))?;

    Ok(())
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    env_logger::init();

    // Create system tray menu
    let tray_menu = SystemTrayMenu::new()
        .add_item(CustomMenuItem::new("show", "Show Window"))
        .add_item(CustomMenuItem::new("search", "Quick Search"))
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("local_mode", "Local Mode"))
        .add_item(CustomMenuItem::new("server_mode", "Server Mode"))
        .add_native_item(tauri::SystemTrayMenuItem::Separator)
        .add_item(CustomMenuItem::new("quit", "Quit"));

    let system_tray = SystemTray::new().with_menu(tray_menu);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_global_shortcut::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_os::init())
        .system_tray(system_tray)
        .setup(|app| {
            // Initialize app state
            let state = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(async { AppState::new().await.unwrap() });

            app.manage(Arc::new(state));

            info!("AIDocIndexer Desktop started");
            Ok(())
        })
        .on_system_tray_event(|app, event| match event {
            SystemTrayEvent::LeftClick { .. } => {
                if let Some(window) = app.get_window("main") {
                    window.show().unwrap();
                    window.set_focus().unwrap();
                }
            }
            SystemTrayEvent::MenuItemClick { id, .. } => match id.as_str() {
                "show" => {
                    if let Some(window) = app.get_window("main") {
                        window.show().unwrap();
                        window.set_focus().unwrap();
                    }
                }
                "search" => {
                    if let Some(window) = app.get_window("main") {
                        window.show().unwrap();
                        window.set_focus().unwrap();
                        window.emit("open-search", ()).unwrap();
                    }
                }
                "local_mode" => {
                    app.emit_all("set-mode", AppMode::Local).unwrap();
                }
                "server_mode" => {
                    app.emit_all("set-mode", AppMode::Server).unwrap();
                }
                "quit" => {
                    std::process::exit(0);
                }
                _ => {}
            },
            _ => {}
        })
        .on_window_event(|event| {
            if let WindowEvent::CloseRequested { api, .. } = event.event() {
                // Hide window instead of closing (minimize to tray)
                event.window().hide().unwrap();
                api.prevent_close();
            }
        })
        .invoke_handler(tauri::generate_handler![
            get_mode,
            set_mode,
            get_settings,
            update_settings,
            check_ollama,
            check_server,
            get_ollama_models,
            chat_local,
            chat_server,
            search_local,
            search_server,
            index_file,
            upload_file,
            get_documents,
            delete_document,
            add_watch_directory,
            remove_watch_directory,
            get_stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
