import { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAppStore } from './lib/store';
import { Sidebar } from './components/Sidebar';
import { Titlebar } from './components/Titlebar';
import { ChatPage } from './pages/ChatPage';
import { DocumentsPage } from './pages/DocumentsPage';
import { SettingsPage } from './pages/SettingsPage';
import { SearchPage } from './pages/SearchPage';
import { initializeTauri } from './lib/tauri';

function App() {
  const { theme, setOllamaStatus, setServerStatus, mode, setMode } = useAppStore();
  const [initialized, setInitialized] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('Starting up...');

  useEffect(() => {
    // Apply theme
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  useEffect(() => {
    // Initialize Tauri and check connections
    const init = async () => {
      try {
        setLoadingStatus('Initializing...');
        await initializeTauri();

        // Get current mode
        setLoadingStatus('Checking mode...');
        const { invoke } = await import('@tauri-apps/api/core');
        const currentMode = await invoke<string>('get_mode');
        setMode(currentMode as 'local' | 'server');

        // Check Ollama status for local mode
        if (currentMode === 'local') {
          setLoadingStatus('Connecting to Ollama...');
          try {
            const ollamaHealthy = await invoke<boolean>('check_ollama_health');
            setOllamaStatus(ollamaHealthy ? 'connected' : 'disconnected');
          } catch {
            setOllamaStatus('disconnected');
          }
        }

        // Check server status for server mode
        if (currentMode === 'server') {
          setLoadingStatus('Connecting to server...');
          try {
            const serverHealthy = await invoke<boolean>('check_server_health');
            setServerStatus(serverHealthy ? 'connected' : 'disconnected');
          } catch {
            setServerStatus('disconnected');
          }
        }

        setLoadingStatus('Ready!');
        setTimeout(() => setInitialized(true), 300);
      } catch (error) {
        console.error('Failed to initialize:', error);
        setInitialized(true);
      }
    };

    init();
  }, [setOllamaStatus, setServerStatus, setMode, mode]);

  if (!initialized) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="flex flex-col items-center gap-6">
          {/* Logo */}
          <div className="relative">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-2xl shadow-primary/30">
              <span className="text-2xl font-bold text-primary-foreground">AI</span>
            </div>
            {/* Animated ring */}
            <div className="absolute inset-0 w-16 h-16 rounded-2xl border-2 border-primary/30 animate-ping" />
          </div>

          {/* App name */}
          <div className="text-center">
            <h1 className="text-xl font-bold">AIDocumentIndexer</h1>
            <p className="text-sm text-muted-foreground mt-1">Desktop</p>
          </div>

          {/* Loading indicator */}
          <div className="flex flex-col items-center gap-3">
            <div className="flex gap-1.5">
              <span className="w-2 h-2 rounded-full bg-primary/60 typing-dot" />
              <span className="w-2 h-2 rounded-full bg-primary/60 typing-dot" />
              <span className="w-2 h-2 rounded-full bg-primary/60 typing-dot" />
            </div>
            <p className="text-sm text-muted-foreground animate-pulse">
              {loadingStatus}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <BrowserRouter>
      <div className="flex flex-col h-screen bg-background">
        <Titlebar />
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-hidden bg-background">
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/documents" element={<DocumentsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
