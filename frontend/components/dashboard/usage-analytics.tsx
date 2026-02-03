'use client';

import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  FileText,
  MessageSquare,
  Search,
  Users,
  Clock,
  Database,
  Cpu,
  HardDrive,
  Activity,
  Calendar,
  RefreshCw,
} from 'lucide-react';

interface UsageStats {
  // Document stats
  totalDocuments: number;
  documentsThisWeek: number;
  documentsGrowth: number;
  totalChunks: number;
  totalEmbeddings: number;

  // Query stats
  totalQueries: number;
  queriesThisWeek: number;
  queriesGrowth: number;
  averageResponseTime: number;

  // Chat stats
  totalConversations: number;
  conversationsThisWeek: number;
  messagesThisWeek: number;

  // System stats
  storageUsed: number;
  storageLimit: number;
  apiCallsThisMonth: number;
  apiCallsLimit: number;

  // User activity
  activeUsers: number;
  peakHour: number;

  // Time series data
  queryHistory: Array<{ date: string; count: number }>;
  documentHistory: Array<{ date: string; count: number }>;
}

interface UsageAnalyticsDashboardProps {
  /** API endpoint for fetching stats */
  apiEndpoint?: string;
  /** Refresh interval in milliseconds */
  refreshInterval?: number;
  /** CSS class name */
  className?: string;
}

export function UsageAnalyticsDashboard({
  apiEndpoint = '/api/v1/analytics/usage',
  refreshInterval = 60000,
  className = '',
}: UsageAnalyticsDashboardProps) {
  const [stats, setStats] = useState<UsageStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [selectedPeriod, setSelectedPeriod] = useState<'7d' | '30d' | '90d'>('7d');

  const fetchStats = async () => {
    try {
      const response = await fetch(`${apiEndpoint}?period=${selectedPeriod}`);
      if (!response.ok) throw new Error('Failed to fetch stats');
      const data = await response.json();
      setStats(data);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, refreshInterval);
    return () => clearInterval(interval);
  }, [selectedPeriod, refreshInterval]);

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatBytes = (bytes: number): string => {
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(1)} GB`;
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} B`;
  };

  const formatDuration = (ms: number): string => {
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
    return `${ms.toFixed(0)}ms`;
  };

  const formatPercentage = (value: number): string => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  };

  if (isLoading && !stats) {
    return (
      <div className={`flex items-center justify-center p-8 ${className}`}>
        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (error && !stats) {
    return (
      <div className={`flex flex-col items-center justify-center p-8 ${className}`}>
        <Activity className="w-12 h-12 text-muted-foreground mb-4" />
        <p className="text-lg font-medium">{error}</p>
        <button
          onClick={fetchStats}
          className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-lg"
        >
          Retry
        </button>
      </div>
    );
  }

  // Use mock data for demonstration if no real data
  const data = stats || {
    totalDocuments: 1234,
    documentsThisWeek: 89,
    documentsGrowth: 12.5,
    totalChunks: 45678,
    totalEmbeddings: 45678,
    totalQueries: 5678,
    queriesThisWeek: 456,
    queriesGrowth: 8.3,
    averageResponseTime: 234,
    totalConversations: 234,
    conversationsThisWeek: 45,
    messagesThisWeek: 567,
    storageUsed: 2147483648,
    storageLimit: 10737418240,
    apiCallsThisMonth: 12345,
    apiCallsLimit: 50000,
    activeUsers: 23,
    peakHour: 14,
    queryHistory: [
      { date: '2024-01-01', count: 120 },
      { date: '2024-01-02', count: 145 },
      { date: '2024-01-03', count: 98 },
      { date: '2024-01-04', count: 167 },
      { date: '2024-01-05', count: 189 },
      { date: '2024-01-06', count: 134 },
      { date: '2024-01-07', count: 178 },
    ],
    documentHistory: [
      { date: '2024-01-01', count: 12 },
      { date: '2024-01-02', count: 8 },
      { date: '2024-01-03', count: 15 },
      { date: '2024-01-04', count: 23 },
      { date: '2024-01-05', count: 11 },
      { date: '2024-01-06', count: 9 },
      { date: '2024-01-07', count: 14 },
    ],
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Usage Analytics</h2>
          {lastUpdated && (
            <p className="text-sm text-muted-foreground">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </p>
          )}
        </div>

        <div className="flex items-center gap-4">
          {/* Period selector */}
          <div className="flex gap-1 p-1 bg-muted rounded-lg">
            {(['7d', '30d', '90d'] as const).map((period) => (
              <button
                key={period}
                onClick={() => setSelectedPeriod(period)}
                className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                  selectedPeriod === period
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {period === '7d' ? '7 Days' : period === '30d' ? '30 Days' : '90 Days'}
              </button>
            ))}
          </div>

          <button
            onClick={fetchStats}
            disabled={isLoading}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
          >
            <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={FileText}
          label="Total Documents"
          value={formatNumber(data.totalDocuments)}
          change={data.documentsGrowth}
          subLabel={`${data.documentsThisWeek} this week`}
          color="blue"
        />
        <MetricCard
          icon={Search}
          label="Total Queries"
          value={formatNumber(data.totalQueries)}
          change={data.queriesGrowth}
          subLabel={`${data.queriesThisWeek} this week`}
          color="purple"
        />
        <MetricCard
          icon={MessageSquare}
          label="Conversations"
          value={formatNumber(data.totalConversations)}
          subLabel={`${data.messagesThisWeek} messages this week`}
          color="green"
        />
        <MetricCard
          icon={Clock}
          label="Avg Response Time"
          value={formatDuration(data.averageResponseTime)}
          subLabel="across all queries"
          color="orange"
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Query history chart */}
        <div className="bg-card rounded-lg border border-border p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">Query Volume</h3>
            <BarChart3 className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="h-48">
            <SimpleBarChart
              data={data.queryHistory}
              color="hsl(var(--primary))"
            />
          </div>
        </div>

        {/* Document uploads chart */}
        <div className="bg-card rounded-lg border border-border p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">Document Uploads</h3>
            <TrendingUp className="w-5 h-5 text-muted-foreground" />
          </div>
          <div className="h-48">
            <SimpleBarChart
              data={data.documentHistory}
              color="hsl(142 76% 36%)"
            />
          </div>
        </div>
      </div>

      {/* System stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Storage usage */}
        <div className="bg-card rounded-lg border border-border p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-blue-500/10">
              <HardDrive className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Storage Used</p>
              <p className="text-lg font-semibold">
                {formatBytes(data.storageUsed)} / {formatBytes(data.storageLimit)}
              </p>
            </div>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${(data.storageUsed / data.storageLimit) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {((data.storageUsed / data.storageLimit) * 100).toFixed(1)}% used
          </p>
        </div>

        {/* API calls */}
        <div className="bg-card rounded-lg border border-border p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Cpu className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">API Calls This Month</p>
              <p className="text-lg font-semibold">
                {formatNumber(data.apiCallsThisMonth)} / {formatNumber(data.apiCallsLimit)}
              </p>
            </div>
          </div>
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-500 transition-all duration-500"
              style={{ width: `${(data.apiCallsThisMonth / data.apiCallsLimit) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            {((data.apiCallsThisMonth / data.apiCallsLimit) * 100).toFixed(1)}% used
          </p>
        </div>

        {/* Active users */}
        <div className="bg-card rounded-lg border border-border p-4">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-green-500/10">
              <Users className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Active Users</p>
              <p className="text-lg font-semibold">{data.activeUsers}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="w-4 h-4" />
            <span>Peak hour: {data.peakHour}:00</span>
          </div>
        </div>
      </div>

      {/* Additional stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={Database} label="Total Chunks" value={formatNumber(data.totalChunks)} />
        <StatCard icon={Activity} label="Embeddings" value={formatNumber(data.totalEmbeddings)} />
        <StatCard
          icon={MessageSquare}
          label="Messages This Week"
          value={formatNumber(data.messagesThisWeek)}
        />
        <StatCard
          icon={Calendar}
          label="Conversations This Week"
          value={formatNumber(data.conversationsThisWeek)}
        />
      </div>
    </div>
  );
}

// Metric card component
function MetricCard({
  icon: Icon,
  label,
  value,
  change,
  subLabel,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  change?: number;
  subLabel?: string;
  color: 'blue' | 'purple' | 'green' | 'orange';
}) {
  const colorClasses = {
    blue: 'bg-blue-500/10 text-blue-500',
    purple: 'bg-purple-500/10 text-purple-500',
    green: 'bg-green-500/10 text-green-500',
    orange: 'bg-orange-500/10 text-orange-500',
  };

  return (
    <div className="bg-card rounded-lg border border-border p-4">
      <div className="flex items-start justify-between">
        <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        {change !== undefined && (
          <div
            className={`flex items-center gap-1 text-sm ${
              change >= 0 ? 'text-green-500' : 'text-red-500'
            }`}
          >
            {change >= 0 ? (
              <TrendingUp className="w-4 h-4" />
            ) : (
              <TrendingDown className="w-4 h-4" />
            )}
            <span>{Math.abs(change).toFixed(1)}%</span>
          </div>
        )}
      </div>
      <div className="mt-3">
        <p className="text-2xl font-bold">{value}</p>
        <p className="text-sm text-muted-foreground">{label}</p>
        {subLabel && <p className="text-xs text-muted-foreground mt-1">{subLabel}</p>}
      </div>
    </div>
  );
}

// Simple stat card
function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
}) {
  return (
    <div className="bg-card rounded-lg border border-border p-4 flex items-center gap-3">
      <Icon className="w-5 h-5 text-muted-foreground" />
      <div>
        <p className="text-lg font-semibold">{value}</p>
        <p className="text-xs text-muted-foreground">{label}</p>
      </div>
    </div>
  );
}

// Simple bar chart component (no external deps)
function SimpleBarChart({
  data,
  color,
}: {
  data: Array<{ date: string; count: number }>;
  color: string;
}) {
  const maxValue = Math.max(...data.map((d) => d.count));

  return (
    <div className="flex items-end justify-between h-full gap-1">
      {data.map((item, index) => (
        <div
          key={index}
          className="flex-1 flex flex-col items-center gap-1"
        >
          <div className="w-full flex items-end justify-center h-36">
            <div
              className="w-full max-w-8 rounded-t transition-all duration-300 hover:opacity-80"
              style={{
                height: `${(item.count / maxValue) * 100}%`,
                backgroundColor: color,
                minHeight: '4px',
              }}
              title={`${item.count}`}
            />
          </div>
          <span className="text-xs text-muted-foreground truncate w-full text-center">
            {new Date(item.date).toLocaleDateString(undefined, { weekday: 'short' })}
          </span>
        </div>
      ))}
    </div>
  );
}

export default UsageAnalyticsDashboard;
