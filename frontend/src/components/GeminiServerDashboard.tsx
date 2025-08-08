import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Activity,
  Brain,
  Database,
  MessageSquare,
  Play,
  Square,
  RefreshCw,
  Zap,
  Settings,
  Terminal,
  FileText,
  Code,
  Layers
} from 'lucide-react';
import { apiService } from '@/services/api';

interface ServerStatus {
  server: string;
  status: string;
  authentication: string;
  model: string;
  sessions: {
    total: number;
    with_keep_alive: number;
  };
  react_engine: {
    type: string;
    available: boolean;
    project_root: string;
    tools_registered: number;
    redis_integration: boolean;
    rag_integration: boolean;
    rust_extensions: boolean;
  };
  capabilities: Record<string, boolean>;
}

interface GeminiSession {
  session_id: string;
  created_at: string;
  last_activity: string;
  conversation_length: number;
  active_tasks: number;
}

interface TaskRequest {
  task_type: 'analysis' | 'code_generation' | 'documentation' | 'consolidation';
  instruction: string;
  target_path?: string;
}

interface WebSocketMessage {
  type: string;
  message?: string;
  result?: any;
  timestamp: string;
  react_step?: {
    type: string;
    description: string;
    tool_name?: string;
  };
}

export function GeminiServerDashboard() {
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [currentSession, setCurrentSession] = useState<GeminiSession | null>(null);
  const [sessions, setSessions] = useState<GeminiSession[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [wsMessages, setWsMessages] = useState<WebSocketMessage[]>([]);
  const [taskRequest, setTaskRequest] = useState<TaskRequest>({
    task_type: 'analysis',
    instruction: '',
    target_path: ''
  });
  const [isExecutingTask, setIsExecutingTask] = useState(false);
  const [loading, setLoading] = useState(true);

  // Load initial data
  const loadData = useCallback(async () => {
    try {
      setLoading(true);

      // Load server status
      const statusResponse = await apiService.getServerStatus();
      if (statusResponse.success) {
        setServerStatus(statusResponse.data);
      }

      // Load current session
      const sessionResponse = await apiService.getCurrentSession();
      if (sessionResponse.success) {
        setCurrentSession(sessionResponse.data);
      }

      // Load all sessions
      const sessionsResponse = await apiService.listActiveAgents();
      if (sessionsResponse.success) {
        const sessionData = sessionsResponse.data.map(agent => ({
          session_id: agent.agent_id,
          created_at: agent.created_at,
          last_activity: agent.last_activity,
          conversation_length: agent.steps_total,
          active_tasks: agent.status === 'running' ? 1 : 0
        }));
        setSessions(sessionData);
      }
    } catch (error) {
      console.error('Failed to load data:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    apiService.connectWebSocket(
      (message: WebSocketMessage) => {
        setWsMessages(prev => [message, ...prev.slice(0, 49)]); // Keep last 50 messages

        if (message.type === 'react_step') {
          console.log('ReAct step:', message.react_step);
        }
      },
      (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      }
    );
    setIsConnected(true);
  }, []);

  const disconnectWebSocket = useCallback(() => {
    apiService.disconnectWebSocket();
    setIsConnected(false);
  }, []);

  // Task execution
  const executeTask = async () => {
    if (!taskRequest.instruction.trim()) return;

    setIsExecutingTask(true);
    try {
      if (taskRequest.task_type === 'consolidation') {
        await apiService.requestConsolidation(taskRequest.instruction, taskRequest.target_path);
      } else {
        // For other task types, use the agent invoke method
        const agentTypeMap = {
          'analysis': 'workspace-analyzer' as const,
          'code_generation': 'code-generator' as const,
          'documentation': 'documentation-generator' as const
        };

        await apiService.invokeAgent(agentTypeMap[taskRequest.task_type], {
          instruction: taskRequest.instruction,
          target_path: taskRequest.target_path
        });
      }

      // Refresh data after task
      await loadData();

      // Clear form
      setTaskRequest({
        task_type: 'analysis',
        instruction: '',
        target_path: ''
      });
    } catch (error) {
      console.error('Task execution failed:', error);
    } finally {
      setIsExecutingTask(false);
    }
  };

  // Create new session
  const createNewSession = async () => {
    try {
      const response = await apiService.createSession();
      if (response.success) {
        await loadData();
      }
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  };

  // Switch to a different session
  const switchToSession = async (sessionId: string) => {
    try {
      await apiService.switchSession(sessionId);
      await loadData();

      // Reconnect WebSocket with new session
      if (isConnected) {
        disconnectWebSocket();
        setTimeout(connectWebSocket, 1000);
      }
    } catch (error) {
      console.error('Failed to switch session:', error);
    }
  };

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading Gemini Server status...</span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Gemini Server Dashboard</h1>
          <p className="text-neutral-400">
            Integrated AI agent execution with context management and caching
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button onClick={loadData} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
          {isConnected ? (
            <Button onClick={disconnectWebSocket} variant="destructive" size="sm">
              <Square className="h-4 w-4 mr-2" />
              Disconnect
            </Button>
          ) : (
            <Button onClick={connectWebSocket} variant="default" size="sm">
              <Play className="h-4 w-4 mr-2" />
              Connect WebSocket
            </Button>
          )}
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Server Status</CardTitle>
            <Activity className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {serverStatus?.status === 'operational' ? 'Online' : 'Offline'}
            </div>
            <p className="text-xs text-neutral-400">
              {serverStatus?.model || 'Unknown model'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Sessions</CardTitle>
            <MessageSquare className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{serverStatus?.sessions.total || 0}</div>
            <p className="text-xs text-neutral-400">
              {serverStatus?.sessions.with_keep_alive || 0} with keep-alive
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ReAct Engine</CardTitle>
            <Brain className="h-4 w-4 text-purple-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {serverStatus?.react_engine.available ? 'Ready' : 'Unavailable'}
            </div>
            <p className="text-xs text-neutral-400">
              {serverStatus?.react_engine.tools_registered || 0} tools registered
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Integrations</CardTitle>
            <Database className="h-4 w-4 text-yellow-400" />
          </CardHeader>
          <CardContent>
            <div className="space-y-1">
              <Badge variant={serverStatus?.react_engine.redis_integration ? "default" : "secondary"}>
                Redis {serverStatus?.react_engine.redis_integration ? '✓' : '✗'}
              </Badge>
              <Badge variant={serverStatus?.react_engine.rag_integration ? "default" : "secondary"}>
                RAG {serverStatus?.react_engine.rag_integration ? '✓' : '✗'}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="execution" className="space-y-4">
        <TabsList>
          <TabsTrigger value="execution">Task Execution</TabsTrigger>
          <TabsTrigger value="sessions">Session Management</TabsTrigger>
          <TabsTrigger value="monitoring">Real-time Monitoring</TabsTrigger>
          <TabsTrigger value="capabilities">Server Capabilities</TabsTrigger>
        </TabsList>

        {/* Task Execution Tab */}
        <TabsContent value="execution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Terminal className="h-5 w-5 mr-2" />
                Execute Task
              </CardTitle>
              <CardDescription>
                Send tasks to the Gemini Server with intelligent ReAct processing
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Task Type</label>
                  <select
                    className="w-full mt-1 p-2 border rounded-md bg-neutral-800 border-neutral-600"
                    value={taskRequest.task_type}
                    onChange={(e) => setTaskRequest(prev => ({
                      ...prev,
                      task_type: e.target.value as TaskRequest['task_type']
                    }))}
                  >
                    <option value="analysis">Analysis</option>
                    <option value="code_generation">Code Generation</option>
                    <option value="documentation">Documentation</option>
                    <option value="consolidation">Consolidation</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Target Path (optional)</label>
                  <input
                    type="text"
                    className="w-full mt-1 p-2 border rounded-md bg-neutral-800 border-neutral-600"
                    placeholder="/path/to/target"
                    value={taskRequest.target_path || ''}
                    onChange={(e) => setTaskRequest(prev => ({
                      ...prev,
                      target_path: e.target.value
                    }))}
                  />
                </div>
              </div>
              <div>
                <label className="text-sm font-medium">Instruction</label>
                <textarea
                  className="w-full mt-1 p-3 border rounded-md bg-neutral-800 border-neutral-600 h-24"
                  placeholder="Describe what you want the AI to do..."
                  value={taskRequest.instruction}
                  onChange={(e) => setTaskRequest(prev => ({
                    ...prev,
                    instruction: e.target.value
                  }))}
                />
              </div>
              <Button
                onClick={executeTask}
                disabled={isExecutingTask || !taskRequest.instruction.trim()}
                className="w-full"
              >
                {isExecutingTask ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Executing Task...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Execute Task
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Session Management Tab */}
        <TabsContent value="sessions" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Session Management</h3>
            <Button onClick={createNewSession} size="sm">
              <MessageSquare className="h-4 w-4 mr-2" />
              New Session
            </Button>
          </div>

          <div className="grid gap-4">
            {currentSession && (
              <Card className="border-blue-500">
                <CardHeader>
                  <CardTitle className="text-sm">Current Session</CardTitle>
                  <CardDescription>{currentSession.session_id}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Created:</span>
                    <span>{new Date(currentSession.created_at).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Last Activity:</span>
                    <span>{new Date(currentSession.last_activity).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Messages:</span>
                    <span>{currentSession.conversation_length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Active Tasks:</span>
                    <Badge variant={currentSession.active_tasks > 0 ? "default" : "secondary"}>
                      {currentSession.active_tasks}
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            )}

            {sessions.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Available Sessions</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-64">
                    <div className="space-y-2">
                      {sessions.map((session) => (
                        <div
                          key={session.session_id}
                          className="flex items-center justify-between p-3 border rounded-md hover:bg-neutral-800 cursor-pointer"
                          onClick={() => switchToSession(session.session_id)}
                        >
                          <div>
                            <div className="text-sm font-medium">{session.session_id}</div>
                            <div className="text-xs text-neutral-400">
                              {session.conversation_length} messages
                            </div>
                          </div>
                          <div className="text-xs text-neutral-400">
                            {new Date(session.last_activity).toLocaleDateString()}
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* Real-time Monitoring Tab */}
        <TabsContent value="monitoring" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Activity className="h-5 w-5 mr-2" />
                Real-time Activity
                <Badge className="ml-2" variant={isConnected ? "default" : "secondary"}>
                  {isConnected ? 'Connected' : 'Disconnected'}
                </Badge>
              </CardTitle>
              <CardDescription>
                Live updates from the Gemini Server and ReAct engine
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-64">
                {wsMessages.length === 0 ? (
                  <div className="text-center text-neutral-400 py-8">
                    {isConnected ? 'No messages yet...' : 'Connect WebSocket to see live updates'}
                  </div>
                ) : (
                  <div className="space-y-2">
                    {wsMessages.map((message, index) => (
                      <div key={index} className="p-3 border rounded-md bg-neutral-800">
                        <div className="flex items-center justify-between mb-2">
                          <Badge variant="outline">{message.type}</Badge>
                          <span className="text-xs text-neutral-400">
                            {new Date(message.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        {message.message && (
                          <div className="text-sm">{message.message}</div>
                        )}
                        {message.react_step && (
                          <div className="text-sm space-y-1">
                            <div className="font-medium">{message.react_step.type}</div>
                            <div className="text-neutral-400">{message.react_step.description}</div>
                            {message.react_step.tool_name && (
                              <Badge variant="secondary" className="text-xs">
                                {message.react_step.tool_name}
                              </Badge>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Server Capabilities Tab */}
        <TabsContent value="capabilities" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="h-5 w-5 mr-2" />
                  Core Capabilities
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {serverStatus?.capabilities && Object.entries(serverStatus.capabilities).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                      <Badge variant={value ? "default" : "secondary"}>
                        {value ? '✓' : '✗'}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Layers className="h-5 w-5 mr-2" />
                  Technical Details
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Authentication:</span>
                    <span>{serverStatus?.authentication}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Engine Type:</span>
                    <span>{serverStatus?.react_engine.type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Tools:</span>
                    <span>{serverStatus?.react_engine.tools_registered}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Rust Extensions:</span>
                    <Badge variant={serverStatus?.react_engine.rust_extensions ? "default" : "secondary"}>
                      {serverStatus?.react_engine.rust_extensions ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
