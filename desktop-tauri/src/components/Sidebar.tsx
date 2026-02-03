import { NavLink } from 'react-router-dom';
import {
  MessageSquare,
  Search,
  FileText,
  Settings,
  ChevronLeft,
  ChevronRight,
  Wifi,
  WifiOff,
  Server,
  HardDrive,
  Sparkles,
  FolderOpen,
} from 'lucide-react';
import { useAppStore } from '../lib/store';
import { cn } from '../lib/utils';

const navItems = [
  {
    to: '/chat',
    icon: MessageSquare,
    label: 'Chat',
    description: 'AI conversations',
    color: 'text-blue-500',
    bgColor: 'bg-blue-500/10',
  },
  {
    to: '/search',
    icon: Search,
    label: 'Search',
    description: 'Find documents',
    color: 'text-purple-500',
    bgColor: 'bg-purple-500/10',
  },
  {
    to: '/documents',
    icon: FolderOpen,
    label: 'Documents',
    description: 'Manage files',
    color: 'text-amber-500',
    bgColor: 'bg-amber-500/10',
  },
  {
    to: '/settings',
    icon: Settings,
    label: 'Settings',
    description: 'Preferences',
    color: 'text-slate-500',
    bgColor: 'bg-slate-500/10',
  },
];

export function Sidebar() {
  const {
    mode,
    ollamaStatus,
    serverStatus,
    sidebarCollapsed,
    setSidebarCollapsed,
  } = useAppStore();

  const isConnected =
    mode === 'local'
      ? ollamaStatus === 'connected'
      : serverStatus === 'connected';

  const connectionLabel = mode === 'local' ? 'Ollama' : 'Server';

  return (
    <aside
      className={cn(
        'flex flex-col glass border-r border-border/50 transition-all duration-300 ease-out',
        sidebarCollapsed ? 'w-[68px]' : 'w-60'
      )}
    >
      {/* Mode indicator */}
      <div className="p-3 border-b border-border/50">
        <div
          className={cn(
            'mode-badge transition-all duration-300',
            sidebarCollapsed ? 'justify-center px-2' : ''
          )}
        >
          <div className={cn(
            'p-1.5 rounded-lg',
            mode === 'local' ? 'bg-emerald-500/20' : 'bg-blue-500/20'
          )}>
            {mode === 'local' ? (
              <HardDrive className="w-4 h-4 text-emerald-500" />
            ) : (
              <Server className="w-4 h-4 text-blue-500" />
            )}
          </div>
          {!sidebarCollapsed && (
            <div className="flex flex-col slide-in-left">
              <span className="text-sm font-semibold">
                {mode === 'local' ? 'Local Mode' : 'Server Mode'}
              </span>
              <span className="text-[10px] text-muted-foreground">
                {mode === 'local' ? 'Offline processing' : 'Cloud connected'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1.5 overflow-y-auto">
        {navItems.map(({ to, icon: Icon, label, description, color, bgColor }, index) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'nav-item group',
                isActive
                  ? 'nav-item-active'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
                sidebarCollapsed && 'justify-center px-2'
              )
            }
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {({ isActive }) => (
              <>
                <div className={cn(
                  'p-2 rounded-lg transition-colors',
                  isActive ? 'bg-primary-foreground/20' : bgColor,
                  'group-hover:scale-110 transition-transform'
                )}>
                  <Icon className={cn(
                    'w-4 h-4 flex-shrink-0',
                    isActive ? 'text-primary-foreground' : color
                  )} />
                </div>
                {!sidebarCollapsed && (
                  <div className="flex flex-col min-w-0">
                    <span className="font-medium truncate">{label}</span>
                    <span className={cn(
                      'text-[10px] truncate',
                      isActive ? 'text-primary-foreground/70' : 'text-muted-foreground'
                    )}>
                      {description}
                    </span>
                  </div>
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* AI Status Badge */}
      {!sidebarCollapsed && (
        <div className="px-3 py-2">
          <div className="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-primary/5 to-purple-500/5 border border-primary/10">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-xs font-medium">AI Ready</span>
          </div>
        </div>
      )}

      {/* Connection status */}
      <div className="p-3 border-t border-border/50">
        <div
          className={cn(
            'flex items-center gap-2 p-2 rounded-lg bg-muted/50 transition-all',
            sidebarCollapsed && 'justify-center'
          )}
        >
          <div className="relative">
            <div className={cn(
              'status-dot',
              isConnected ? 'connected' : 'disconnected'
            )} />
          </div>
          {!sidebarCollapsed && (
            <div className="flex flex-col min-w-0">
              <span className="text-xs font-medium truncate">
                {connectionLabel}
              </span>
              <span className={cn(
                'text-[10px]',
                isConnected ? 'text-green-500' : 'text-red-500'
              )}>
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Collapse button */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className={cn(
          'flex items-center gap-2 p-3 border-t border-border/50',
          'hover:bg-muted/50 transition-all duration-200',
          'text-muted-foreground hover:text-foreground',
          sidebarCollapsed && 'justify-center'
        )}
      >
        <div className={cn(
          'p-1.5 rounded-lg bg-muted/50 transition-transform duration-300',
          sidebarCollapsed ? 'rotate-0' : 'rotate-180'
        )}>
          <ChevronLeft className="w-4 h-4" />
        </div>
        {!sidebarCollapsed && (
          <span className="text-xs font-medium">Collapse</span>
        )}
      </button>
    </aside>
  );
}
