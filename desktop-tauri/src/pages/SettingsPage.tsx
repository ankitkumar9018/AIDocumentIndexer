import { useState, useEffect } from 'react';
import {
  Sun,
  Moon,
  Server,
  HardDrive,
  Loader2,
  Check,
  AlertCircle,
  RefreshCw,
  Settings,
  Palette,
  Cpu,
  Database,
  Zap,
  Save,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { useAppStore } from '../lib/store';
import {
  setMode as setTauriMode,
  checkOllamaHealth,
  checkServerHealth,
  getOllamaModels,
  updateSettings as updateTauriSettings,
  type OllamaModel,
} from '../lib/tauri';
import { cn } from '../lib/utils';

export function SettingsPage() {
  const {
    theme,
    setTheme,
    mode,
    setMode,
    settings,
    updateSettings,
    ollamaStatus,
    setOllamaStatus,
    serverStatus,
    setServerStatus,
  } = useAppStore();

  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  useEffect(() => {
    if (mode === 'local') {
      loadOllamaModels();
    }
  }, [mode]);

  const loadOllamaModels = async () => {
    setIsLoadingModels(true);
    try {
      const models = await getOllamaModels();
      setOllamaModels(models);
    } catch (error) {
      console.error('Failed to load Ollama models:', error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const handleModeChange = async (newMode: 'local' | 'server') => {
    try {
      await setTauriMode(newMode);
      setMode(newMode);

      // Check connection for new mode
      if (newMode === 'local') {
        setOllamaStatus('checking');
        const healthy = await checkOllamaHealth();
        setOllamaStatus(healthy ? 'connected' : 'disconnected');
        if (healthy) loadOllamaModels();
      } else {
        setServerStatus('checking');
        const healthy = await checkServerHealth();
        setServerStatus(healthy ? 'connected' : 'disconnected');
      }
    } catch (error) {
      console.error('Failed to change mode:', error);
    }
  };

  const handleCheckConnection = async () => {
    if (mode === 'local') {
      setOllamaStatus('checking');
      try {
        const healthy = await checkOllamaHealth();
        setOllamaStatus(healthy ? 'connected' : 'disconnected');
      } catch {
        setOllamaStatus('disconnected');
      }
    } else {
      setServerStatus('checking');
      try {
        const healthy = await checkServerHealth();
        setServerStatus(healthy ? 'connected' : 'disconnected');
      } catch {
        setServerStatus('disconnected');
      }
    }
  };

  const handleSaveSettings = async () => {
    setIsSaving(true);
    setSaveSuccess(false);
    try {
      await updateTauriSettings({
        server_url: settings.serverUrl,
        ollama_model: settings.ollamaModel,
        embedding_model: settings.embeddingModel,
        chunk_size: settings.chunkSize,
        chunk_overlap: settings.chunkOverlap,
      });
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2000);
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const connectionStatus = mode === 'local' ? ollamaStatus : serverStatus;

  return (
    <div className="flex flex-col h-full page-enter">
      {/* Header */}
      <header className="flex items-center gap-3 px-4 py-3 border-b border-border/50 glass">
        <div className="p-2 rounded-xl bg-gradient-to-br from-slate-500/20 to-slate-500/5">
          <Settings className="w-5 h-5 text-slate-500" />
        </div>
        <div>
          <h1 className="text-lg font-semibold">Settings</h1>
          <p className="text-xs text-muted-foreground">
            Configure your preferences
          </p>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Appearance Section */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Palette className="w-4 h-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold">Appearance</h2>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setTheme('light')}
              className={cn(
                'flex items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200',
                theme === 'light'
                  ? 'border-primary bg-primary/5 shadow-md shadow-primary/10'
                  : 'border-border/50 bg-card hover:border-primary/50 hover:bg-muted/50'
              )}
            >
              <div className={cn(
                'p-2 rounded-lg',
                theme === 'light' ? 'bg-amber-500/20' : 'bg-muted'
              )}>
                <Sun className={cn(
                  'w-5 h-5',
                  theme === 'light' ? 'text-amber-500' : 'text-muted-foreground'
                )} />
              </div>
              <div className="text-left">
                <span className="font-medium">Light</span>
                <p className="text-xs text-muted-foreground">Bright theme</p>
              </div>
            </button>
            <button
              onClick={() => setTheme('dark')}
              className={cn(
                'flex items-center gap-3 p-4 rounded-xl border-2 transition-all duration-200',
                theme === 'dark'
                  ? 'border-primary bg-primary/5 shadow-md shadow-primary/10'
                  : 'border-border/50 bg-card hover:border-primary/50 hover:bg-muted/50'
              )}
            >
              <div className={cn(
                'p-2 rounded-lg',
                theme === 'dark' ? 'bg-indigo-500/20' : 'bg-muted'
              )}>
                <Moon className={cn(
                  'w-5 h-5',
                  theme === 'dark' ? 'text-indigo-400' : 'text-muted-foreground'
                )} />
              </div>
              <div className="text-left">
                <span className="font-medium">Dark</span>
                <p className="text-xs text-muted-foreground">Easy on eyes</p>
              </div>
            </button>
          </div>
        </section>

        {/* Mode Section */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold">Operation Mode</h2>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => handleModeChange('local')}
              className={cn(
                'flex flex-col items-start p-4 rounded-xl border-2 transition-all duration-200',
                mode === 'local'
                  ? 'border-emerald-500 bg-emerald-500/5 shadow-md shadow-emerald-500/10'
                  : 'border-border/50 bg-card hover:border-emerald-500/50 hover:bg-muted/50'
              )}
            >
              <div className={cn(
                'p-2 rounded-lg mb-2',
                mode === 'local' ? 'bg-emerald-500/20' : 'bg-muted'
              )}>
                <HardDrive className={cn(
                  'w-5 h-5',
                  mode === 'local' ? 'text-emerald-500' : 'text-muted-foreground'
                )} />
              </div>
              <span className="font-medium">Local Mode</span>
              <p className="text-xs text-muted-foreground mt-0.5">
                Offline with Ollama
              </p>
            </button>
            <button
              onClick={() => handleModeChange('server')}
              className={cn(
                'flex flex-col items-start p-4 rounded-xl border-2 transition-all duration-200',
                mode === 'server'
                  ? 'border-blue-500 bg-blue-500/5 shadow-md shadow-blue-500/10'
                  : 'border-border/50 bg-card hover:border-blue-500/50 hover:bg-muted/50'
              )}
            >
              <div className={cn(
                'p-2 rounded-lg mb-2',
                mode === 'server' ? 'bg-blue-500/20' : 'bg-muted'
              )}>
                <Server className={cn(
                  'w-5 h-5',
                  mode === 'server' ? 'text-blue-500' : 'text-muted-foreground'
                )} />
              </div>
              <span className="font-medium">Server Mode</span>
              <p className="text-xs text-muted-foreground mt-0.5">
                Connect to backend
              </p>
            </button>
          </div>

          {/* Connection status */}
          <div className="flex items-center justify-between p-3 rounded-xl bg-muted/50 border border-border/30">
            <div className="flex items-center gap-3">
              <div className={cn(
                'p-2 rounded-lg',
                connectionStatus === 'connected' ? 'bg-green-500/20' : 'bg-red-500/20'
              )}>
                {connectionStatus === 'checking' ? (
                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                ) : connectionStatus === 'connected' ? (
                  <Wifi className="w-4 h-4 text-green-500" />
                ) : (
                  <WifiOff className="w-4 h-4 text-red-500" />
                )}
              </div>
              <div>
                <span className="text-sm font-medium">
                  {mode === 'local' ? 'Ollama' : 'Server'}
                </span>
                <p className={cn(
                  'text-xs',
                  connectionStatus === 'connected' ? 'text-green-500' : 'text-red-500'
                )}>
                  {connectionStatus === 'checking'
                    ? 'Checking connection...'
                    : connectionStatus === 'connected'
                      ? 'Connected'
                      : 'Disconnected'}
                </p>
              </div>
            </div>
            <button
              onClick={handleCheckConnection}
              disabled={connectionStatus === 'checking'}
              className="p-2 rounded-lg hover:bg-background/50 transition-colors disabled:opacity-50"
            >
              <RefreshCw
                className={cn(
                  'w-4 h-4 text-muted-foreground',
                  connectionStatus === 'checking' && 'animate-spin'
                )}
              />
            </button>
          </div>
        </section>

        {/* Server Settings (only in server mode) */}
        {mode === 'server' && (
          <section className="space-y-3 slide-in-up">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold">Server Connection</h2>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                  Server URL
                </label>
                <input
                  type="text"
                  value={settings.serverUrl}
                  onChange={(e) =>
                    updateSettings({ serverUrl: e.target.value })
                  }
                  placeholder="http://localhost:8000"
                  className="w-full px-3 py-2.5 rounded-xl input-modern"
                />
              </div>
            </div>
          </section>
        )}

        {/* Local Mode Settings (only in local mode) */}
        {mode === 'local' && (
          <section className="space-y-3 slide-in-up">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-muted-foreground" />
              <h2 className="text-sm font-semibold">Ollama Configuration</h2>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                  Chat Model
                </label>
                <div className="flex gap-2">
                  <select
                    value={settings.ollamaModel}
                    onChange={(e) =>
                      updateSettings({ ollamaModel: e.target.value })
                    }
                    className="flex-1 px-3 py-2.5 rounded-xl input-modern"
                  >
                    {ollamaModels.length === 0 ? (
                      <option value={settings.ollamaModel}>
                        {settings.ollamaModel}
                      </option>
                    ) : (
                      ollamaModels.map((model) => (
                        <option key={model.name} value={model.name}>
                          {model.name}
                        </option>
                      ))
                    )}
                  </select>
                  <button
                    onClick={loadOllamaModels}
                    disabled={isLoadingModels}
                    className="p-2.5 rounded-xl bg-muted/50 hover:bg-muted transition-colors disabled:opacity-50"
                  >
                    <RefreshCw
                      className={cn(
                        'w-4 h-4',
                        isLoadingModels && 'animate-spin'
                      )}
                    />
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                  Embedding Model
                </label>
                <input
                  type="text"
                  value={settings.embeddingModel}
                  onChange={(e) =>
                    updateSettings({ embeddingModel: e.target.value })
                  }
                  placeholder="nomic-embed-text"
                  className="w-full px-3 py-2.5 rounded-xl input-modern"
                />
              </div>
            </div>
          </section>
        )}

        {/* Processing Settings */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold">Document Processing</h2>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                Chunk Size
              </label>
              <input
                type="number"
                value={settings.chunkSize}
                onChange={(e) =>
                  updateSettings({ chunkSize: parseInt(e.target.value) || 512 })
                }
                min={100}
                max={2000}
                className="w-full px-3 py-2.5 rounded-xl input-modern"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1.5">
                Chunk Overlap
              </label>
              <input
                type="number"
                value={settings.chunkOverlap}
                onChange={(e) =>
                  updateSettings({
                    chunkOverlap: parseInt(e.target.value) || 50,
                  })
                }
                min={0}
                max={500}
                className="w-full px-3 py-2.5 rounded-xl input-modern"
              />
            </div>
          </div>
          <p className="text-[10px] text-muted-foreground">
            Smaller chunks = more precise, larger chunks = more context
          </p>
        </section>
      </div>

      {/* Sticky Save Button */}
      <div className="p-4 border-t border-border/50 glass">
        <button
          onClick={handleSaveSettings}
          disabled={isSaving}
          className={cn(
            'w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl font-medium',
            'transition-all duration-200',
            saveSuccess
              ? 'bg-green-500 text-white'
              : 'btn-gradient',
            'disabled:opacity-50 disabled:cursor-not-allowed'
          )}
        >
          {isSaving ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Saving...</span>
            </>
          ) : saveSuccess ? (
            <>
              <Check className="w-4 h-4" />
              <span>Saved!</span>
            </>
          ) : (
            <>
              <Save className="w-4 h-4" />
              <span>Save Settings</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
