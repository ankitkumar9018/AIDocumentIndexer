//! File watcher for auto-indexing
//!
//! Watches directories for file changes and triggers indexing.

use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::thread;
use std::time::Duration;

use log::{error, info};
use notify::{Config, Event, RecommendedWatcher, RecursiveMode, Watcher};

pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    stop_tx: Sender<()>,
}

impl FileWatcher {
    pub fn new<F>(directories: Vec<PathBuf>, callback: F) -> Self
    where
        F: Fn(PathBuf) + Send + 'static,
    {
        let (stop_tx, stop_rx) = channel();
        let (event_tx, event_rx) = channel();

        // Create debounced event handler
        let callback = std::sync::Arc::new(callback);
        let callback_clone = callback.clone();

        thread::spawn(move || {
            let mut pending_events: std::collections::HashMap<PathBuf, std::time::Instant> =
                std::collections::HashMap::new();
            let debounce_duration = Duration::from_millis(500);

            loop {
                // Check for stop signal
                if stop_rx.try_recv().is_ok() {
                    break;
                }

                // Process events
                while let Ok(path) = event_rx.try_recv() {
                    pending_events.insert(path, std::time::Instant::now());
                }

                // Execute debounced callbacks
                let now = std::time::Instant::now();
                let mut to_process = Vec::new();

                pending_events.retain(|path, time| {
                    if now.duration_since(*time) >= debounce_duration {
                        to_process.push(path.clone());
                        false
                    } else {
                        true
                    }
                });

                for path in to_process {
                    callback_clone(path);
                }

                thread::sleep(Duration::from_millis(100));
            }
        });

        // Create watcher
        let event_tx_clone = event_tx.clone();
        let mut watcher =
            notify::recommended_watcher(move |res: Result<Event, notify::Error>| match res {
                Ok(event) => {
                    if matches!(
                        event.kind,
                        notify::EventKind::Create(_)
                            | notify::EventKind::Modify(_)
                            | notify::EventKind::Remove(_)
                    ) {
                        for path in event.paths {
                            if is_supported_file(&path) {
                                let _ = event_tx_clone.send(path);
                            }
                        }
                    }
                }
                Err(e) => error!("File watch error: {:?}", e),
            })
            .expect("Failed to create file watcher");

        // Watch directories
        for dir in &directories {
            if let Err(e) = watcher.watch(dir, RecursiveMode::Recursive) {
                error!("Failed to watch directory {:?}: {}", dir, e);
            } else {
                info!("Watching directory: {:?}", dir);
            }
        }

        Self {
            _watcher: watcher,
            stop_tx,
        }
    }

    pub fn stop(&mut self) {
        let _ = self.stop_tx.send(());
    }
}

/// Check if file type is supported for indexing
fn is_supported_file(path: &PathBuf) -> bool {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    matches!(
        extension.as_str(),
        "txt" | "md" | "markdown" | "pdf" | "docx" | "doc" | "rtf" | "html" | "htm" | "json"
            | "csv" | "xml" | "yaml" | "yml" | "toml" | "rs" | "py" | "js" | "ts" | "tsx" | "jsx"
            | "java" | "go" | "rb" | "php" | "c" | "cpp" | "h" | "hpp" | "swift" | "kt"
    )
}
