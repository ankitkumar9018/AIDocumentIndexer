//! AIDocumentIndexer Desktop Library
//!
//! Core library for the Tauri desktop application.

pub mod database;
pub mod embeddings;
pub mod file_watcher;
pub mod ollama;
pub mod server_client;

pub use database::LocalDatabase;
pub use file_watcher::FileWatcher;
pub use ollama::OllamaClient;
pub use server_client::ServerClient;
