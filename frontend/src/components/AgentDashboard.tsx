import React, { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Play,
  Square,
  RefreshCw,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Zap,
  Brain,
  Database
} from 'lucide-react';
import {
  AgentType,
  AgentStatus,
  AgentStatusResponse,
  AgentInvokeRequest,
  AgentStatusEvent
} from '@/types/api';
import { apiService } from '@/services/api';
import { getWebSocketService } from '@/services/websocket';

interface AgentCardProps {
  agent: AgentStatusResponse;
  onInvoke: (agentType: AgentType, request: AgentInvokeRequest) => void;
  onCancel: (agentId: string) => void;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent, onInvoke, onCancel }) => {
  const getStatusIcon = (status: AgentStatus) => {
    switch (status) {
      case AgentStatus.RUNNING:
        return <Activity className="w-4 h-4 text-green-500 animate-pulse" />;
      case AgentStatus.ERROR:
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case AgentStatus.COMPLETED:
        return <CheckCircle className="w-4 h-4 text-blue-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: AgentStatus) => {
    switch (status) {
      case AgentStatus.RUNNING:
        return 'bg-green-500/20 text-green-300 border-green-500/30';
      case AgentStatus.ERROR:
        return 'bg-red-500/20 text-red-300 border-red-500/30';
      case AgentStatus.COMPLETED:
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
      default:
        return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  };

  const getAgentIcon = (type: AgentType) => {
    switch (type) {
      case AgentType.MASTER_ARCHITECT:
        return <Brain className="w-5 h-5" />;
      case AgentType.CODE_GENERATOR_PRO:
        return <Zap className="w-5 h-5" />;
      case AgentType.MEMORY_RAG:
        return <Database className="w-5 h-5" />;
    }
  };

  const formatDuration = (startTime: string) => {
    const start = new Date(startTime);
    const now = new Date();
    const diff = now.getTime() - start.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <Card className="p-4 border-neutral-700 bg-neutral-800/50 hover:bg-neutral-800 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          {getAgentIcon(agent.agentType)}
          <h3 className="font-semibold text-neutral-100">
            {agent.agentType.replace('_', ' ').toLowerCase().replace(/\b\w/g, l => l.toUpperCase())}
          </h3>
        </div>
        <Badge className={`${getStatusColor(agent.status)} border`}>
          <div className="flex items-center gap-1">
            {getStatusIcon(agent.status)}
            {agent.status}
          </div>
        </Badge>
      </div>

      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-neutral-400">Agent ID:</span>
          <span className="text-neutral-300 font-mono text-xs">{agent.agentId}</span>
        </div>

        <div className="flex justify-between text-sm">
          <span className="text-neutral-400">Runtime:</span>
          <span className="text-neutral-300">{formatDuration(agent.startTime)}</span>
        </div>

        {agent.progress !== undefined && (
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-neutral-400">Progress:</span>
              <span className="text-neutral-300">{Math.round(agent.progress)}%</span>
            </div>
            <div className="w-full bg-neutral-700 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${agent.progress}%` }}
              />
            </div>
          </div>
        )}

        {agent.currentTask && (
          <div className="text-sm">
            <span className="text-neutral-400">Current Task:</span>
            <p className="text-neutral-300 mt-1 text-xs">{agent.currentTask}</p>
          </div>
        )}

        {agent.error && (
          <div className="text-sm">
            <span className="text-red-400">Error:</span>
            <p className="text-red-300 mt-1 text-xs">{agent.error}</p>
          </div>
        )}

        {agent.estimatedCompletion && (
          <div className="flex justify-between text-sm">
            <span className="text-neutral-400">ETA:</span>
            <span className="text-neutral-300">
              {new Date(agent.estimatedCompletion).toLocaleTimeString()}
            </span>
          </div>
        )}
      </div>

      <div className="flex gap-2">
        {agent.status === AgentStatus.IDLE && (
          <Button
            size="sm"
            onClick={() => onInvoke(agent.agentType, { prompt: 'Quick test invocation' })}
            className="flex-1 bg-blue-600 hover:bg-blue-700"
          >
            <Play className="w-3 h-3 mr-1" />
            Start
          </Button>
        )}

        {agent.status === AgentStatus.RUNNING && (
          <Button
            size="sm"
            variant="outline"
            onClick={() => onCancel(agent.agentId)}
            className="flex-1 border-red-500/30 text-red-300 hover:bg-red-500/20"
          >
            <Square className="w-3 h-3 mr-1" />
            Cancel
          </Button>
        )}

        <Button
          size="sm"
          variant="outline"
          className="border-neutral-600 text-neutral-300 hover:bg-neutral-700"
        >
          View Details
        </Button>
      </div>
    </Card>
  );
};

export const AgentDashboard: React.FC = () => {
  const [agents, setAgents] = useState<AgentStatusResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('all');

  const loadAgents = useCallback(async () => {
    try {
      setLoading(true);
      const response = await apiService.listActiveAgents();
      if (response.success) {
        setAgents(response.data);
      } else {
        setError(response.error?.message || 'Failed to load agents');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleInvokeAgent = useCallback(async (agentType: AgentType, request: AgentInvokeRequest) => {
    try {
      const response = await apiService.invokeAgent(agentType, request);
      if (response.success) {
        // Refresh the agents list
        loadAgents();
      } else {
        setError(response.error?.message || 'Failed to invoke agent');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [loadAgents]);

  const handleCancelAgent = useCallback(async (agentId: string) => {
    try {
      const response = await apiService.cancelAgent(agentId);
      if (response.success) {
        // Refresh the agents list
        loadAgents();
      } else {
        setError(response.error?.message || 'Failed to cancel agent');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, [loadAgents]);

  // Set up WebSocket for real-time updates
  useEffect(() => {
    const wsBaseUrl = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsBaseUrl}//${window.location.host}/api/v1/agents/stream`;

    try {
      const ws = getWebSocketService(wsUrl);

      const handleAgentStatus = (data: AgentStatusEvent) => {
        setAgents(prev => {
          const index = prev.findIndex(a => a.agentId === data.agentId);
          if (index >= 0) {
            const updated = [...prev];
            updated[index] = {
              ...updated[index],
              status: data.status,
              progress: data.progress,
              currentTask: data.currentTask,
            };
            return updated;
          } else {
            // New agent, add to list
            return [...prev, {
              agentId: data.agentId,
              agentType: data.agentType,
              status: data.status,
              progress: data.progress,
              currentTask: data.currentTask,
              startTime: new Date().toISOString(),
            }];
          }
        });
      };

      ws.on('agent_status', handleAgentStatus);
      ws.connect().catch(console.error);

      return () => {
        ws.off('agent_status', handleAgentStatus);
        ws.disconnect();
      };
    } catch (err) {
      console.warn('WebSocket not available, falling back to polling');
    }
  }, []);

  // Initial load and periodic refresh
  useEffect(() => {
    loadAgents();
    const interval = setInterval(loadAgents, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, [loadAgents]);

  const filteredAgents = agents.filter(agent => {
    if (activeTab === 'all') return true;
    if (activeTab === 'running') return agent.status === AgentStatus.RUNNING;
    if (activeTab === 'completed') return agent.status === AgentStatus.COMPLETED;
    if (activeTab === 'error') return agent.status === AgentStatus.ERROR;
    return true;
  });

  const getTabCounts = () => {
    return {
      all: agents.length,
      running: agents.filter(a => a.status === AgentStatus.RUNNING).length,
      completed: agents.filter(a => a.status === AgentStatus.COMPLETED).length,
      error: agents.filter(a => a.status === AgentStatus.ERROR).length,
    };
  };

  const tabCounts = getTabCounts();

  if (loading && agents.length === 0) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="w-8 h-8 animate-spin text-neutral-400" />
          <span className="ml-2 text-neutral-400">Loading agents...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Agent Dashboard</h1>
          <p className="text-neutral-400">Monitor and manage AI agents in real-time</p>
        </div>
        <Button onClick={loadAgents} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {error && (
        <Card className="p-4 border-red-500/30 bg-red-500/10">
          <div className="flex items-center gap-2 text-red-300">
            <AlertCircle className="w-4 h-4" />
            <span>{error}</span>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setError(null)}
              className="ml-auto text-red-300 hover:text-red-200"
            >
              Dismiss
            </Button>
          </div>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4 bg-neutral-800">
          <TabsTrigger value="all" className="data-[state=active]:bg-neutral-700">
            All ({tabCounts.all})
          </TabsTrigger>
          <TabsTrigger value="running" className="data-[state=active]:bg-neutral-700">
            Running ({tabCounts.running})
          </TabsTrigger>
          <TabsTrigger value="completed" className="data-[state=active]:bg-neutral-700">
            Completed ({tabCounts.completed})
          </TabsTrigger>
          <TabsTrigger value="error" className="data-[state=active]:bg-neutral-700">
            Errors ({tabCounts.error})
          </TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-6">
          {filteredAgents.length === 0 ? (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Activity className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No agents found
              </h3>
              <p className="text-neutral-400">
                {activeTab === 'all'
                  ? 'No agents are currently active'
                  : `No agents with status "${activeTab}"`
                }
              </p>
            </Card>
          ) : (
            <ScrollArea className="h-[600px]">
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {filteredAgents.map((agent) => (
                  <AgentCard
                    key={agent.agentId}
                    agent={agent}
                    onInvoke={handleInvokeAgent}
                    onCancel={handleCancelAgent}
                  />
                ))}
              </div>
            </ScrollArea>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};
