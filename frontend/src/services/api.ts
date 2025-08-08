import {
  AgentType,
  AgentInvokeRequest,
  AgentInvokeResponse,
  AgentStatusResponse,
  ContextAnalyzeRequest,
  ContextAnalyzeResponse,
  ContextSearchRequest,
  ContextSearchResponse,
  CodeGenerationRequest,
  CodeGenerationResponse,
  MemoryEmbedRequest,
  MemoryEmbedResponse,
  MemorySearchRequest,
  MemorySearchResponse,
  KnowledgeGraph,
  ApiResponse,
  ProjectContext
} from '@/types/api';

// Gemini Server types for integration
interface GeminiTaskRequest {
  task_type: string;
  instruction: string;
  target_path?: string;
  session_id?: string;
  options?: Record<string, any>;
  priority?: 'low' | 'normal' | 'high' | 'urgent';
  timeout_seconds?: number;
  cache_key?: string;
}

interface GeminiTaskResponse {
  status: 'completed' | 'failed' | 'pending';
  task_type: string;
  session_id: string;
  instruction: string;
  target_path?: string;
  timestamp: string;
  react_engine: {
    success: boolean;
    total_time: number;
    steps_executed: number;
    step_details: Array<{
      type: string;
      description: string;
      tool_name?: string;
      success?: boolean;
      timestamp: string;
    }>;
  };
  result: any;
  error?: string;
}

interface GeminiSession {
  session_id: string;
  created_at: string;
  last_activity: string;
  conversation_length: number;
  active_tasks: number;
}

class ApiService {
  private baseUrl: string;
  private geminiServerUrl: string;
  private currentSessionId?: string;
  private wsConnection?: WebSocket;

  constructor(baseUrl: string = '/api/v1', geminiServerUrl: string = 'http://127.0.0.1:8100') {
    this.baseUrl = baseUrl;
    this.geminiServerUrl = geminiServerUrl;
    this.initializeSession();
  }

  private async initializeSession(): Promise<void> {
    try {
      // Try to restore session from localStorage
      const savedSessionId = localStorage.getItem('gemini_session_id');
      if (savedSessionId) {
        // Verify session exists on server
        const response = await fetch(`${this.geminiServerUrl}/sessions/${savedSessionId}`);
        if (response.ok) {
          this.currentSessionId = savedSessionId;
          return;
        }
      }

      // Create new session if none exists or saved session is invalid
      await this.createNewSession();
    } catch (error) {
      console.warn('Failed to initialize session:', error);
      // Continue without session - server will create one
    }
  }

  private async createNewSession(): Promise<string> {
    try {
      // Session will be created automatically by server when we send first task
      // For now, generate a client-side session ID
      const sessionId = `web_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      this.currentSessionId = sessionId;
      localStorage.setItem('gemini_session_id', sessionId);
      return sessionId;
    } catch (error) {
      console.error('Failed to create session:', error);
      throw error;
    }
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    useGeminiServer: boolean = false
  ): Promise<ApiResponse<T>> {
    const baseUrl = useGeminiServer ? this.geminiServerUrl : this.baseUrl;
    const url = `${baseUrl}${endpoint}`;
    const defaultHeaders = {
      'Content-Type': 'application/json',
    };

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error?.message || `Request failed with status ${response.status}`);
      }

      return data;
    } catch (error) {
      console.error(`API request failed: ${url}`, error);
      throw error;
    }
  }

  // Gemini Server integration methods
  private async requestGeminiServer<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const response = await this.request<T>(endpoint, options, true);
    return response.data || response;
  }

  private async executeGeminiTask(taskRequest: GeminiTaskRequest): Promise<GeminiTaskResponse> {
    // Ensure session ID is set
    if (!taskRequest.session_id && this.currentSessionId) {
      taskRequest.session_id = this.currentSessionId;
    }

    const response = await this.requestGeminiServer<GeminiTaskResponse>('/task', {
      method: 'POST',
      body: JSON.stringify(taskRequest),
    });

    // Update session ID if server created a new one
    if (response.session_id && response.session_id !== this.currentSessionId) {
      this.currentSessionId = response.session_id;
      localStorage.setItem('gemini_session_id', response.session_id);
    }

    return response;
  }

  // WebSocket connection for real-time updates
  public connectWebSocket(onMessage?: (data: any) => void, onError?: (error: Event) => void): void {
    if (this.wsConnection) {
      this.wsConnection.close();
    }

    const wsUrl = `ws://127.0.0.1:8100/ws?session_id=${this.currentSessionId || ''}`;
    this.wsConnection = new WebSocket(wsUrl);

    this.wsConnection.onopen = () => {
      console.log('WebSocket connected to Gemini Server');
    };

    this.wsConnection.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (onMessage) {
          onMessage(data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.wsConnection.onerror = (error) => {
      console.error('WebSocket error:', error);
      if (onError) {
        onError(error);
      }
    };

    this.wsConnection.onclose = () => {
      console.log('WebSocket disconnected');
      // Auto-reconnect after 5 seconds
      setTimeout(() => {
        if (!this.wsConnection || this.wsConnection.readyState === WebSocket.CLOSED) {
          this.connectWebSocket(onMessage, onError);
        }
      }, 5000);
    };
  }

  public disconnectWebSocket(): void {
    if (this.wsConnection) {
      this.wsConnection.close();
      this.wsConnection = undefined;
    }
  }

  // Agent Management (using Gemini Server)
  async invokeAgent(
    agentType: AgentType,
    request: AgentInvokeRequest
  ): Promise<ApiResponse<AgentInvokeResponse>> {
    // Map agent types to task types for Gemini Server
    const taskTypeMap: Record<AgentType, string> = {
      'code-reviewer': 'analysis',
      'workspace-analyzer': 'analysis',
      'documentation-generator': 'documentation',
      'code-generator': 'code_generation',
      'master-architect': 'analysis'
    };

    const taskRequest: GeminiTaskRequest = {
      task_type: taskTypeMap[agentType] || 'analysis',
      instruction: request.instruction || `Execute ${agentType} task`,
      target_path: request.target_path,
      session_id: this.currentSessionId,
      options: {
        agent_type: agentType,
        ...request.options
      },
      priority: request.priority as any || 'normal',
      timeout_seconds: request.timeout_seconds || 300
    };

    try {
      const geminiResponse = await this.executeGeminiTask(taskRequest);

      // Convert Gemini response to expected format
      const agentResponse: AgentInvokeResponse = {
        agent_id: geminiResponse.session_id,
        status: geminiResponse.status === 'completed' ? 'completed' :
                geminiResponse.status === 'failed' ? 'failed' : 'running',
        result: geminiResponse.result,
        execution_time: geminiResponse.react_engine.total_time,
        steps_executed: geminiResponse.react_engine.steps_executed,
        error: geminiResponse.error,
        context: {
          session_id: geminiResponse.session_id,
          task_type: geminiResponse.task_type,
          react_steps: geminiResponse.react_engine.step_details
        }
      };

      return { data: agentResponse, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async getAgentStatus(agentId: string): Promise<ApiResponse<AgentStatusResponse>> {
    // In the context of Gemini Server, agent_id is actually session_id
    try {
      const session = await this.requestGeminiServer<any>(`/sessions/${agentId}`);

      const statusResponse: AgentStatusResponse = {
        agent_id: agentId,
        status: session.keep_alive_active ? 'running' : 'idle',
        created_at: session.session_data.created_at,
        last_activity: session.session_data.last_activity,
        progress: 100, // Completed tasks
        current_step: session.session_data.active_tasks.length > 0 ? 'processing' : 'idle',
        steps_total: session.session_data.conversation_length,
        result: session.session_data.context_cache,
        error: null
      };

      return { data: statusResponse, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async cancelAgent(agentId: string): Promise<ApiResponse<void>> {
    try {
      await this.requestGeminiServer(`/sessions/${agentId}`, {
        method: 'DELETE',
      });
      return { data: undefined, success: true };
    } catch (error: any) {
      return {
        data: undefined,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async listActiveAgents(): Promise<ApiResponse<AgentStatusResponse[]>> {
    try {
      const sessionsResponse = await this.requestGeminiServer<{
        total_sessions: number;
        sessions: Record<string, GeminiSession>;
      }>('/sessions');

      const agents: AgentStatusResponse[] = Object.entries(sessionsResponse.sessions).map(
        ([sessionId, session]) => ({
          agent_id: sessionId,
          status: session.active_tasks > 0 ? 'running' : 'idle',
          created_at: session.created_at,
          last_activity: session.last_activity,
          progress: 100,
          current_step: session.active_tasks > 0 ? 'processing' : 'idle',
          steps_total: session.conversation_length,
          result: null,
          error: null
        })
      );

      return { data: agents, success: true };
    } catch (error: any) {
      return {
        data: [],
        success: false,
        error: { message: error.message }
      };
    }
  }

  // Context Analysis (using Gemini Server)
  async analyzeContext(request: ContextAnalyzeRequest): Promise<ApiResponse<ContextAnalyzeResponse>> {
    const taskRequest: GeminiTaskRequest = {
      task_type: 'analysis',
      instruction: `Analyze project context: ${request.instruction || 'Perform comprehensive context analysis'}`,
      target_path: request.project_path,
      session_id: this.currentSessionId,
      options: {
        analysis_type: 'context',
        depth: request.depth || 3,
        include_dependencies: request.include_dependencies !== false,
        ...request.options
      },
      cache_key: `context_analysis_${request.project_path}_${JSON.stringify(request.options)}`
    };

    try {
      const geminiResponse = await this.executeGeminiTask(taskRequest);

      const contextResponse: ContextAnalyzeResponse = {
        context_id: geminiResponse.session_id,
        project_path: request.project_path,
        analysis: geminiResponse.result,
        created_at: geminiResponse.timestamp,
        metadata: {
          execution_time: geminiResponse.react_engine.total_time,
          steps_executed: geminiResponse.react_engine.steps_executed,
          react_steps: geminiResponse.react_engine.step_details
        }
      };

      return { data: contextResponse, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async getContext(contextId: string): Promise<ApiResponse<ProjectContext>> {
    // Context is stored in sessions in Gemini Server
    try {
      const session = await this.requestGeminiServer<any>(`/sessions/${contextId}`);

      const context: ProjectContext = {
        context_id: contextId,
        project_path: session.session_data.orchestrator_state.target_path || '',
        created_at: session.session_data.created_at,
        updated_at: session.session_data.last_activity,
        analysis: session.session_data.context_cache,
        metadata: {
          conversation_length: session.session_data.conversation_history.length,
          active_tasks: session.session_data.active_tasks.length
        }
      };

      return { data: context, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async searchContext(request: ContextSearchRequest): Promise<ApiResponse<ContextSearchResponse>> {
    // Use session-based context search via Gemini Server
    const taskRequest: GeminiTaskRequest = {
      task_type: 'analysis',
      instruction: `Search context for: ${request.query}. Provide relevant results with confidence scores.`,
      session_id: this.currentSessionId,
      options: {
        search_type: 'context',
        query: request.query,
        max_results: request.max_results || 10,
        filters: request.filters
      }
    };

    try {
      const geminiResponse = await this.executeGeminiTask(taskRequest);

      const searchResponse: ContextSearchResponse = {
        query: request.query,
        results: Array.isArray(geminiResponse.result) ? geminiResponse.result : [geminiResponse.result],
        total_results: Array.isArray(geminiResponse.result) ? geminiResponse.result.length : 1,
        execution_time: geminiResponse.react_engine.total_time
      };

      return { data: searchResponse, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async listContexts(): Promise<ApiResponse<ProjectContext[]>> {
    try {
      const sessionsResponse = await this.requestGeminiServer<{
        total_sessions: number;
        sessions: Record<string, GeminiSession>;
      }>('/sessions');

      const contexts: ProjectContext[] = Object.entries(sessionsResponse.sessions).map(
        ([sessionId, session]) => ({
          context_id: sessionId,
          project_path: '', // Not directly available from session data
          created_at: session.created_at,
          updated_at: session.last_activity,
          analysis: {}, // Summary would require individual session fetch
          metadata: {
            conversation_length: session.conversation_length,
            active_tasks: session.active_tasks
          }
        })
      );

      return { data: contexts, success: true };
    } catch (error: any) {
      return {
        data: [],
        success: false,
        error: { message: error.message }
      };
    }
  }

  async deleteContext(contextId: string): Promise<ApiResponse<void>> {
    // Delete session in Gemini Server
    return this.cancelAgent(contextId); // Reuse the session deletion logic
  }

  // Code Generation (using Gemini Server)
  async generateCode(request: CodeGenerationRequest): Promise<ApiResponse<CodeGenerationResponse>> {
    const taskRequest: GeminiTaskRequest = {
      task_type: 'code_generation',
      instruction: request.specification,
      target_path: request.target_path,
      session_id: this.currentSessionId,
      options: {
        language: request.language,
        style: request.style,
        include_tests: request.include_tests !== false,
        include_docs: request.include_docs !== false,
        ...request.options
      },
      cache_key: `code_gen_${request.language}_${JSON.stringify(request.options)}`
    };

    try {
      const geminiResponse = await this.executeGeminiTask(taskRequest);

      const codeResponse: CodeGenerationResponse = {
        request_id: geminiResponse.session_id,
        status: geminiResponse.status === 'completed' ? 'completed' :
                geminiResponse.status === 'failed' ? 'failed' : 'running',
        specification: request.specification,
        language: request.language,
        generated_code: geminiResponse.result,
        created_at: geminiResponse.timestamp,
        completed_at: geminiResponse.status === 'completed' ? geminiResponse.timestamp : undefined,
        execution_time: geminiResponse.react_engine.total_time,
        error: geminiResponse.error,
        metadata: {
          steps_executed: geminiResponse.react_engine.steps_executed,
          react_steps: geminiResponse.react_engine.step_details
        }
      };

      return { data: codeResponse, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async getGenerationStatus(requestId: string): Promise<ApiResponse<CodeGenerationResponse>> {
    // In Gemini Server context, requestId is session_id
    return this.getAgentStatus(requestId) as any; // Type coercion for compatibility
  }

  async cancelGeneration(requestId: string): Promise<ApiResponse<void>> {
    return this.cancelAgent(requestId);
  }

  async listGenerations(): Promise<ApiResponse<CodeGenerationResponse[]>> {
    // Map active agents to code generations
    const agentsResponse = await this.listActiveAgents();
    if (!agentsResponse.success) {
      return agentsResponse as any;
    }

    const generations: CodeGenerationResponse[] = agentsResponse.data
      .filter(agent => agent.current_step === 'processing' || agent.status === 'completed')
      .map(agent => ({
        request_id: agent.agent_id,
        status: agent.status as any,
        specification: 'Code generation task', // Generic - actual spec would need session details
        language: 'unknown', // Would need session details
        generated_code: agent.result,
        created_at: agent.created_at,
        completed_at: agent.status === 'completed' ? agent.last_activity : undefined,
        execution_time: 0, // Would need detailed timing
        error: agent.error?.message,
        metadata: {}
      }));

    return { data: generations, success: true };
  }

  // Memory RAG
  async embedMemory(request: MemoryEmbedRequest): Promise<ApiResponse<MemoryEmbedResponse>> {
    return this.request('/memory/embed', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async searchMemory(request: MemorySearchRequest): Promise<ApiResponse<MemorySearchResponse>> {
    return this.request('/memory/search', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getKnowledgeGraph(contextId?: string): Promise<ApiResponse<KnowledgeGraph>> {
    const endpoint = contextId
      ? `/memory/knowledge-graph?contextId=${contextId}`
      : '/memory/knowledge-graph';
    return this.request(endpoint);
  }

  async updateKnowledgeGraph(
    contextId: string,
    update: Partial<KnowledgeGraph>
  ): Promise<ApiResponse<KnowledgeGraph>> {
    return this.request(`/memory/knowledge-graph/${contextId}`, {
      method: 'PATCH',
      body: JSON.stringify(update),
    });
  }

  async deleteMemory(embeddingId: string): Promise<ApiResponse<void>> {
    return this.request(`/memory/${embeddingId}`, {
      method: 'DELETE',
    });
  }

  // File operations
  async uploadFile(file: File, contextId?: string): Promise<ApiResponse<{ fileId: string }>> {
    const formData = new FormData();
    formData.append('file', file);
    if (contextId) {
      formData.append('contextId', contextId);
    }

    return this.request('/files/upload', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for multipart/form-data
    });
  }

  async downloadFile(fileId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/files/${fileId}/download`);
    if (!response.ok) {
      throw new Error(`File download failed: ${response.statusText}`);
    }
    return response.blob();
  }

  // Health and metrics (using Gemini Server)
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    try {
      const health = await this.requestGeminiServer<{
        status: string;
        model_available: boolean;
        active_sessions: number;
        active_tasks: number;
        timestamp: string;
      }>('/health');

      return {
        data: {
          status: health.status,
          timestamp: health.timestamp
        },
        success: true
      };
    } catch (error: any) {
      return {
        data: { status: 'error', timestamp: new Date().toISOString() },
        success: false,
        error: { message: error.message }
      };
    }
  }

  async getMetrics(): Promise<ApiResponse<Record<string, any>>> {
    try {
      const [health, status, reactStatus] = await Promise.all([
        this.requestGeminiServer<any>('/health'),
        this.requestGeminiServer<any>('/status'),
        this.requestGeminiServer<any>('/react/status')
      ]);

      const metrics = {
        server: {
          status: health.status,
          active_sessions: health.active_sessions,
          active_tasks: health.active_tasks,
          model_available: health.model_available
        },
        react_engine: {
          type: status.react_engine.type,
          available: status.react_engine.available,
          tools_registered: status.react_engine.tools_registered,
          redis_integration: status.react_engine.redis_integration,
          rag_integration: status.react_engine.rag_integration,
          rust_extensions: status.react_engine.rust_extensions
        },
        capabilities: status.capabilities,
        performance: reactStatus.performance || {},
        timestamp: health.timestamp
      };

      return { data: metrics, success: true };
    } catch (error: any) {
      return {
        data: {},
        success: false,
        error: { message: error.message }
      };
    }
  }

  // Session Management (Gemini Server specific)
  async getCurrentSession(): Promise<ApiResponse<GeminiSession>> {
    if (!this.currentSessionId) {
      return {
        data: null as any,
        success: false,
        error: { message: 'No active session' }
      };
    }

    try {
      const session = await this.requestGeminiServer<{
        session_id: string;
        session_data: any;
        keep_alive_active: boolean;
      }>(`/sessions/${this.currentSessionId}`);

      const geminiSession: GeminiSession = {
        session_id: session.session_id,
        created_at: session.session_data.created_at,
        last_activity: session.session_data.last_activity,
        conversation_length: session.session_data.conversation_history.length,
        active_tasks: session.session_data.active_tasks.length
      };

      return { data: geminiSession, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async createSession(): Promise<ApiResponse<{ session_id: string }>> {
    try {
      const sessionId = await this.createNewSession();
      return {
        data: { session_id: sessionId },
        success: true
      };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async switchSession(sessionId: string): Promise<ApiResponse<void>> {
    try {
      // Verify session exists
      const response = await fetch(`${this.geminiServerUrl}/sessions/${sessionId}`);
      if (!response.ok) {
        throw new Error('Session not found');
      }

      // Switch to the session
      this.currentSessionId = sessionId;
      localStorage.setItem('gemini_session_id', sessionId);

      // Reconnect WebSocket if active
      if (this.wsConnection) {
        this.disconnectWebSocket();
        this.connectWebSocket();
      }

      return { data: undefined, success: true };
    } catch (error: any) {
      return {
        data: undefined,
        success: false,
        error: { message: error.message }
      };
    }
  }

  // Enhanced methods for direct Gemini Server features
  async getServerStatus(): Promise<ApiResponse<any>> {
    try {
      const status = await this.requestGeminiServer<any>('/status');
      return { data: status, success: true };
    } catch (error: any) {
      return {
        data: null,
        success: false,
        error: { message: error.message }
      };
    }
  }

  async getReactEngineStatus(): Promise<ApiResponse<any>> {
    try {
      const status = await this.requestGeminiServer<any>('/react/status');
      return { data: status, success: true };
    } catch (error: any) {
      return {
        data: null,
        success: false,
        error: { message: error.message }
      };
    }
  }

  // Consolidation task (Gemini Server specific)
  async requestConsolidation(instruction: string, targetPath?: string): Promise<ApiResponse<GeminiTaskResponse>> {
    const taskRequest: GeminiTaskRequest = {
      task_type: 'consolidation',
      instruction,
      target_path: targetPath,
      session_id: this.currentSessionId,
      priority: 'high',
      timeout_seconds: 600, // Consolidation can take longer
      cache_key: `consolidation_${targetPath}_${Date.now()}`
    };

    try {
      const response = await this.executeGeminiTask(taskRequest);
      return { data: response, success: true };
    } catch (error: any) {
      return {
        data: null as any,
        success: false,
        error: { message: error.message }
      };
    }
  }
}

// Singleton instance
export const apiService = new ApiService();
export default apiService;
