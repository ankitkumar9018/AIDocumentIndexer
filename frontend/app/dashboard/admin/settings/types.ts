/**
 * Shared TypeScript types for Settings components
 */

export interface Setting {
  key: string;
  value: string;
  value_type: string;
  category: string;
  description: string;
}

export interface LLMProvider {
  id: string;
  name: string;
  provider_type: string;
  api_base_url: string | null;
  organization_id: string | null;
  is_active: boolean;
  is_default: boolean;
  default_chat_model: string | null;
  default_embedding_model: string | null;
  settings: Record<string, unknown> | null;
  api_key_masked: string | null;
  has_api_key: boolean;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderType {
  name: string;
  fields: string[];
  required_fields: string[];
  chat_models: string[];
  embedding_models: string[];
  default_chat_model: string | null;
  default_embedding_model: string | null;
  default_api_base_url?: string;
}

export interface DatabaseConnection {
  id: string;
  name: string;
  connection_type: string;
  host?: string;
  port?: number;
  database?: string;
  username?: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface DatabaseConnectionType {
  name: string;
  fields: string[];
  required_fields: string[];
  default_port?: number;
}

export interface SystemHealth {
  status: string;
  version: string;
  uptime: number;
  database: {
    status: string;
    type: string;
    size?: string;
  };
  cache: {
    status: string;
    entries?: number;
  };
  storage: {
    status: string;
    used?: string;
    available?: string;
  };
}

export interface DatabaseInfo {
  type: string;
  size: string;
  path?: string;
  tables: number;
  documents: number;
  chunks: number;
  entities: number;
}

export interface LLMOperation {
  operation: string;
  provider_id: string | null;
  model: string | null;
  temperature: number | null;
  max_tokens: number | null;
}

export interface ProviderHealth {
  provider_id: string;
  provider_name: string;
  status: string;
  last_check: string;
  response_time_ms: number | null;
  error_message: string | null;
  circuit_state: string;
  failure_count: number;
}

export interface LLMUsageSummary {
  total_requests: number;
  total_tokens: number;
  total_cost: number;
  providers: Array<{
    provider: string;
    requests: number;
    tokens: number;
    cost: number;
  }>;
}
