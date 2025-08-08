// API Types for Gemini Agent System

// Agent Types
export enum AgentType {
  MASTER_ARCHITECT = 'MASTER_ARCHITECT',
  CODE_GENERATOR_PRO = 'CODE_GENERATOR_PRO',
  MEMORY_RAG = 'MEMORY_RAG'
}

export enum LanguageType {
  PYTHON = 'python',
  RUST = 'rust',
  TYPESCRIPT = 'typescript',
  JAVASCRIPT = 'javascript',
  GO = 'go',
  JAVA = 'java'
}

export enum AgentStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  ERROR = 'error',
  COMPLETED = 'completed'
}

// Agent Management
export interface AgentInvokeRequest {
  prompt: string;
  context?: ProjectContext;
  config?: Record<string, any>;
}

export interface AgentInvokeResponse {
  agentId: string;
  status: AgentStatus;
  result?: any;
  error?: string;
  timestamp: string;
}

export interface AgentStatusResponse {
  agentId: string;
  agentType: AgentType;
  status: AgentStatus;
  progress?: number;
  currentTask?: string;
  error?: string;
  startTime: string;
  estimatedCompletion?: string;
}

// Project Context
export interface ProjectStructure {
  rootPath: string;
  files: FileNode[];
  totalFiles: number;
  totalDirectories: number;
}

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  lastModified?: string;
  children?: FileNode[];
}

export interface ProjectDependency {
  name: string;
  version: string;
  type: 'runtime' | 'dev' | 'peer';
  language: LanguageType;
  vulnerabilities?: SecurityVulnerability[];
}

export interface SecurityVulnerability {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  fixedIn?: string;
}

export interface ProjectContext {
  id: string;
  name: string;
  structure: ProjectStructure;
  dependencies: ProjectDependency[];
  language: LanguageType;
  analysisCache: Record<string, any>;
  metadata: {
    createdAt: string;
    updatedAt: string;
    analysisVersion: string;
  };
}

// Context Analysis
export interface ContextAnalyzeRequest {
  projectPath: string;
  includeFiles?: string[];
  excludePatterns?: string[];
  analysisDepth?: number;
}

export interface ContextAnalyzeResponse {
  contextId: string;
  context: ProjectContext;
  insights: ProjectInsight[];
  recommendations: string[];
}

export interface ProjectInsight {
  type: 'architecture' | 'security' | 'performance' | 'maintainability';
  severity: 'info' | 'warning' | 'error';
  title: string;
  description: string;
  location?: {
    file: string;
    line?: number;
    column?: number;
  };
  suggestion?: string;
}

export interface ContextSearchRequest {
  query: string;
  contextId?: string;
  filters?: {
    fileTypes?: string[];
    languages?: LanguageType[];
    dateRange?: {
      start: string;
      end: string;
    };
  };
}

export interface ContextSearchResponse {
  results: SearchResult[];
  total: number;
  took: number;
}

export interface SearchResult {
  contextId: string;
  context: ProjectContext;
  score: number;
  highlights: string[];
}

// Code Generation
export interface CodeGenerationRequest {
  prompt: string;
  language: LanguageType;
  context?: ProjectContext;
  constraints?: {
    maxFiles?: number;
    maxLinesPerFile?: number;
    includeTests?: boolean;
    includeDocumentation?: boolean;
  };
  preferences?: {
    codeStyle?: string;
    framework?: string;
    patterns?: string[];
  };
}

export interface GeneratedFile {
  path: string;
  content: string;
  language: LanguageType;
  purpose: string;
  dependencies?: string[];
}

export interface ProAgentInsight {
  type: 'best_practice' | 'optimization' | 'security' | 'architecture';
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
  implementation?: string;
}

export interface CodeGenerationResponse {
  requestId: string;
  generatedFiles: GeneratedFile[];
  proAgentInsights: ProAgentInsight[];
  executionPlan?: {
    steps: string[];
    estimatedTime: string;
  };
  metadata: {
    totalLines: number;
    totalFiles: number;
    generationTime: number;
  };
}

// Memory RAG
export interface MemoryEmbedRequest {
  content: string;
  metadata: {
    type: 'code' | 'documentation' | 'conversation' | 'file';
    source: string;
    language?: LanguageType;
    context?: Record<string, any>;
  };
  tags?: string[];
}

export interface MemoryEmbedResponse {
  embeddingId: string;
  vector: number[];
  metadata: Record<string, any>;
}

export interface MemorySearchRequest {
  query: string;
  topK?: number;
  threshold?: number;
  filters?: {
    type?: string[];
    language?: LanguageType[];
    tags?: string[];
    dateRange?: {
      start: string;
      end: string;
    };
  };
}

export interface MemorySearchResult {
  embeddingId: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  highlights?: string[];
}

export interface MemorySearchResponse {
  results: MemorySearchResult[];
  total: number;
  took: number;
}

// Knowledge Graph
export interface KnowledgeNode {
  id: string;
  type: 'file' | 'function' | 'class' | 'module' | 'concept';
  label: string;
  properties: Record<string, any>;
  metadata: {
    source: string;
    language?: LanguageType;
    confidence: number;
  };
}

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  type: 'imports' | 'calls' | 'inherits' | 'implements' | 'references' | 'contains';
  properties: Record<string, any>;
  weight: number;
}

export interface KnowledgeGraph {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  metadata: {
    nodeCount: number;
    edgeCount: number;
    lastUpdated: string;
  };
}

// WebSocket Events
export interface WebSocketMessage {
  type: 'agent_status' | 'generation_progress' | 'error' | 'completion';
  data: any;
  timestamp: string;
}

export interface AgentStatusEvent {
  agentId: string;
  agentType: AgentType;
  status: AgentStatus;
  progress?: number;
  currentTask?: string;
  logs?: string[];
}

export interface GenerationProgressEvent {
  requestId: string;
  progress: number;
  currentFile?: string;
  completedFiles: number;
  totalFiles: number;
  insights?: ProAgentInsight[];
}

// Common Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  metadata?: {
    requestId: string;
    timestamp: string;
    processingTime: number;
  };
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
  hasPrevious: boolean;
}
