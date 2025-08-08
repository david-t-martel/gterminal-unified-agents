// MCP Server Management Types

export enum MCPServerStatus {
  RUNNING = 'running',
  STOPPED = 'stopped',
  ERROR = 'error',
  STARTING = 'starting',
  STOPPING = 'stopping',
  UNKNOWN = 'unknown'
}

export enum MCPServerType {
  GEMINI_CODE_REVIEWER = 'gemini-code-reviewer',
  GEMINI_WORKSPACE_ANALYZER = 'gemini-workspace-analyzer',
  GEMINI_MASTER_ARCHITECT = 'gemini-master-architect',
  CLOUD_COST_OPTIMIZER = 'cloud-cost-optimizer',
  GITHUB_OFFICIAL = 'github-official',
  RUST_FS = 'rust-fs',
  RUST_FETCH = 'rust-fetch',
  RUST_LINK = 'rust-link'
}

export enum HealthStatus {
  HEALTHY = 'healthy',
  UNHEALTHY = 'unhealthy',
  DEGRADED = 'degraded',
  UNKNOWN = 'unknown'
}

export enum ValidationResultType {
  PASS = 'pass',
  FAIL = 'fail',
  WARNING = 'warning',
  SKIP = 'skip'
}

// Core MCP Server Interface
export interface MCPServer {
  id: string;
  name: string;
  type: MCPServerType;
  status: MCPServerStatus;
  health: HealthStatus;
  config: MCPServerConfig;
  metrics: MCPServerMetrics;
  lastHealthCheck: string;
  uptime: number;
  description?: string;
  version?: string;
  capabilities: string[];
}

// Configuration Management
export interface MCPServerConfig {
  command: string;
  args: string[];
  env: Record<string, string>;
  cwd?: string;
  timeout?: number;
  retries?: number;
  restartPolicy?: 'always' | 'on-failure' | 'never';
  healthCheck?: {
    enabled: boolean;
    interval: number;
    timeout: number;
    retries: number;
  };
  resources?: {
    maxMemory?: string;
    maxCpu?: string;
  };
  logging?: {
    level: 'debug' | 'info' | 'warn' | 'error';
    file?: string;
    maxSize?: string;
  };
}

// Performance Metrics
export interface MCPServerMetrics {
  requests: {
    total: number;
    successful: number;
    failed: number;
    averageResponseTime: number;
    requestsPerSecond: number;
  };
  resources: {
    cpuUsage: number;
    memoryUsage: number;
    memoryLimit?: number;
    diskUsage?: number;
  };
  errors: {
    total: number;
    recent: MCPError[];
    errorRate: number;
  };
  cache?: {
    hitRate: number;
    missRate: number;
    size: number;
  };
}

export interface MCPError {
  timestamp: string;
  level: 'error' | 'warning' | 'info';
  message: string;
  stack?: string;
  context?: Record<string, any>;
}

// Health Check Results
export interface HealthCheckResult {
  serverId: string;
  status: HealthStatus;
  timestamp: string;
  responseTime: number;
  checks: {
    connectivity: ValidationResult;
    authentication: ValidationResult;
    capabilities: ValidationResult;
    performance: ValidationResult;
  };
  issues?: string[];
  recommendations?: string[];
}

export interface ValidationResult {
  type: ValidationResultType;
  message: string;
  details?: any;
  duration?: number;
}

// Schema Validation
export interface SchemaValidationResult {
  serverId: string;
  isValid: boolean;
  issues: SchemaIssue[];
  fixes: SchemaFix[];
  compatibility: {
    claude: boolean;
    mcpInspector: boolean;
    protocol: string;
  };
}

export interface SchemaIssue {
  type: 'error' | 'warning' | 'info';
  category: 'schema' | 'compatibility' | 'performance' | 'security';
  message: string;
  location?: string;
  suggestion?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

export interface SchemaFix {
  id: string;
  description: string;
  type: 'automatic' | 'manual' | 'suggested';
  impact: 'low' | 'medium' | 'high';
  before: string;
  after: string;
  applied?: boolean;
}

// Testing & Validation
export interface MCPTestSuite {
  serverId: string;
  tests: MCPTest[];
  summary: TestSummary;
  startTime: string;
  endTime?: string;
  duration?: number;
}

export interface MCPTest {
  id: string;
  name: string;
  category: 'connectivity' | 'protocol' | 'performance' | 'security' | 'compliance';
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped';
  result?: ValidationResult;
  duration?: number;
  retries?: number;
}

export interface TestSummary {
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  coverage?: number;
}

// GitHub Integration
export interface GitHubRepository {
  id: string;
  name: string;
  fullName: string;
  private: boolean;
  description?: string;
  language: string;
  stargazersCount: number;
  forksCount: number;
  issuesCount: number;
  updatedAt: string;
  topics: string[];
}

export interface GitHubIssue {
  id: string;
  number: number;
  title: string;
  state: 'open' | 'closed';
  author: string;
  assignees: string[];
  labels: string[];
  createdAt: string;
  updatedAt: string;
  body?: string;
}

export interface GitHubPullRequest {
  id: string;
  number: number;
  title: string;
  state: 'open' | 'closed' | 'merged';
  author: string;
  reviewers: string[];
  createdAt: string;
  updatedAt: string;
  mergeable?: boolean;
  checks?: {
    total: number;
    passed: number;
    failed: number;
  };
}

export interface GitHubSearchRequest {
  query: string;
  type: 'repositories' | 'code' | 'issues' | 'pulls' | 'users';
  filters?: {
    language?: string;
    sort?: string;
    order?: 'asc' | 'desc';
    per_page?: number;
    page?: number;
  };
}

export interface GitHubSearchResponse {
  total_count: number;
  incomplete_results: boolean;
  items: any[];
}

// Real-time Updates
export interface MCPServerEvent {
  type: 'status_change' | 'metrics_update' | 'error' | 'log' | 'test_result';
  serverId: string;
  timestamp: string;
  data: any;
}

export interface MCPStatusChangeEvent {
  serverId: string;
  previousStatus: MCPServerStatus;
  currentStatus: MCPServerStatus;
  reason?: string;
  timestamp: string;
}

export interface MCPMetricsUpdateEvent {
  serverId: string;
  metrics: Partial<MCPServerMetrics>;
  timestamp: string;
}

export interface MCPLogEvent {
  serverId: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: string;
  context?: Record<string, any>;
}

// API Request/Response Types
export interface MCPServerListResponse {
  servers: MCPServer[];
  total: number;
  status: {
    running: number;
    stopped: number;
    error: number;
  };
}

export interface MCPServerActionRequest {
  serverId: string;
  action: 'start' | 'stop' | 'restart' | 'health_check' | 'validate';
  options?: Record<string, any>;
}

export interface MCPServerActionResponse {
  serverId: string;
  action: string;
  success: boolean;
  message?: string;
  result?: any;
  timestamp: string;
}

export interface MCPConfigUpdateRequest {
  serverId: string;
  config: Partial<MCPServerConfig>;
  validateOnly?: boolean;
}

export interface MCPConfigUpdateResponse {
  serverId: string;
  success: boolean;
  validation: SchemaValidationResult;
  changes: string[];
  requiresRestart: boolean;
}

// Monitoring & Analytics
export interface TimeRange {
  start: string;
  end: string;
  granularity?: 'minute' | 'hour' | 'day';
}

export interface MetricsQuery {
  serverIds?: string[];
  metrics: string[];
  timeRange: TimeRange;
  aggregation?: 'avg' | 'sum' | 'max' | 'min';
}

export interface MetricsResponse {
  query: MetricsQuery;
  data: {
    serverId: string;
    metrics: {
      [metricName: string]: {
        timestamps: string[];
        values: number[];
      };
    };
  }[];
}

// Dashboard Configuration
export interface DashboardConfig {
  layout: 'grid' | 'list' | 'compact';
  columns: number;
  refreshInterval: number;
  showMetrics: boolean;
  showLogs: boolean;
  theme: 'light' | 'dark' | 'auto';
  notifications: {
    statusChanges: boolean;
    errors: boolean;
    performance: boolean;
  };
}

export interface UserPreferences {
  dashboard: DashboardConfig;
  defaultServerView: 'overview' | 'metrics' | 'logs' | 'config';
  autoRefresh: boolean;
  compactMode: boolean;
}

// Bulk Operations
export interface BulkOperationRequest {
  serverIds: string[];
  operation: 'start' | 'stop' | 'restart' | 'health_check' | 'validate' | 'update_config';
  options?: Record<string, any>;
}

export interface BulkOperationResponse {
  operation: string;
  results: {
    serverId: string;
    success: boolean;
    message?: string;
    error?: string;
  }[];
  summary: {
    total: number;
    successful: number;
    failed: number;
  };
}

// Export Management
export interface ExportRequest {
  type: 'config' | 'metrics' | 'logs' | 'health_reports' | 'all';
  serverIds?: string[];
  timeRange?: TimeRange;
  format: 'json' | 'csv' | 'yaml';
}

export interface ExportResponse {
  exportId: string;
  downloadUrl: string;
  expiresAt: string;
  fileSize: number;
  format: string;
}
