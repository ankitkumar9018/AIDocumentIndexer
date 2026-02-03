import { useState, useEffect } from 'react';
import { Minus, Square, X, Maximize2, Copy } from 'lucide-react';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { cn } from '../lib/utils';

export function Titlebar() {
  const [isMaximized, setIsMaximized] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    const checkMaximized = async () => {
      const appWindow = getCurrentWindow();
      const maximized = await appWindow.isMaximized();
      setIsMaximized(maximized);
    };

    checkMaximized();

    // Listen for window state changes
    let unlisten: (() => void) | undefined;
    getCurrentWindow()
      .onResized(async () => {
        const maximized = await getCurrentWindow().isMaximized();
        setIsMaximized(maximized);
      })
      .then((fn) => {
        unlisten = fn;
      });

    return () => {
      unlisten?.();
    };
  }, []);

  const handleMinimize = async () => {
    await getCurrentWindow().minimize();
  };

  const handleMaximize = async () => {
    const appWindow = getCurrentWindow();
    if (isMaximized) {
      await appWindow.unmaximize();
    } else {
      await appWindow.maximize();
    }
  };

  const handleClose = async () => {
    await getCurrentWindow().close();
  };

  return (
    <div
      data-tauri-drag-region
      className="flex items-center justify-between h-10 glass border-b border-border/50 px-3 select-none"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* App branding */}
      <div className="flex items-center gap-2.5" data-tauri-drag-region>
        <div className="relative">
          <div className="w-5 h-5 rounded-lg bg-gradient-to-br from-primary to-primary/60 flex items-center justify-center shadow-sm shadow-primary/20">
            <span className="text-[10px] font-bold text-primary-foreground">AI</span>
          </div>
          {/* Subtle glow effect */}
          <div className="absolute inset-0 w-5 h-5 rounded-lg bg-primary/20 blur-sm -z-10" />
        </div>
        <div className="flex flex-col">
          <span className="text-sm font-semibold text-foreground leading-tight">
            AIDocIndexer
          </span>
          <span className="text-[9px] text-muted-foreground leading-tight">
            Desktop
          </span>
        </div>
      </div>

      {/* Window controls - Traffic light style with fade animation */}
      <div
        className={cn(
          'flex items-center gap-1.5 transition-opacity duration-200',
          !isHovered && 'opacity-60'
        )}
      >
        {/* Minimize */}
        <button
          onClick={handleMinimize}
          className={cn(
            'group relative flex items-center justify-center w-7 h-7 rounded-full',
            'bg-amber-500/10 hover:bg-amber-500 transition-all duration-200',
            'hover:shadow-lg hover:shadow-amber-500/20'
          )}
          title="Minimize"
        >
          <Minus className={cn(
            'w-3 h-3 text-amber-600 dark:text-amber-400',
            'group-hover:text-white transition-colors'
          )} />
        </button>

        {/* Maximize */}
        <button
          onClick={handleMaximize}
          className={cn(
            'group relative flex items-center justify-center w-7 h-7 rounded-full',
            'bg-green-500/10 hover:bg-green-500 transition-all duration-200',
            'hover:shadow-lg hover:shadow-green-500/20'
          )}
          title={isMaximized ? 'Restore' : 'Maximize'}
        >
          {isMaximized ? (
            <Copy className={cn(
              'w-3 h-3 text-green-600 dark:text-green-400',
              'group-hover:text-white transition-colors'
            )} />
          ) : (
            <Maximize2 className={cn(
              'w-3 h-3 text-green-600 dark:text-green-400',
              'group-hover:text-white transition-colors'
            )} />
          )}
        </button>

        {/* Close */}
        <button
          onClick={handleClose}
          className={cn(
            'group relative flex items-center justify-center w-7 h-7 rounded-full',
            'bg-red-500/10 hover:bg-red-500 transition-all duration-200',
            'hover:shadow-lg hover:shadow-red-500/20'
          )}
          title="Close"
        >
          <X className={cn(
            'w-3 h-3 text-red-600 dark:text-red-400',
            'group-hover:text-white transition-colors'
          )} />
        </button>
      </div>
    </div>
  );
}
