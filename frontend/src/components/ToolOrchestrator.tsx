import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Activity,
  Play,
  Square,
  RefreshCw,
  Layers,
  Zap,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import { apiService } from '@/services/api';

interface ToolTask {
  id: string;
  tool_name: string;
  parameters: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  error?: string;
  execution_time?: number;
}

interface OrchestrationPlan {
  plan_id: string;
  goal: string;
  mode: 'sequential' | 'parallel' | 'reactive';
  tasks: ToolTask[];
  status: 'created' | 'executing' | 'completed' | 'failed';
  total_execution_time?: number;
  final_result?: any;
}

interface OrchestratorStatus {
  active_plans: number;
  completed_plans: number;
  available_tools: number;
  rust_tools_available: boolean;
  metrics: {
    plans_executed: number;
    tools_called: number;
    avg_execution_time: number;
    success_rate: number;
  };
  tool_registry: string[];
}

export function ToolOrchestrator() {
  const [orchestratorStatus, setOrchestratorStatus] = useState<OrchestratorStatus | null>(null);
  const [activePlans, setActivePlans] = useState<OrchestrationPlan[]>([]);
  const [completedPlans, setCompletedPlans] = useState<OrchestrationPlan[]>([]);
  const [newPlan, setNewPlan] = useState({
    goal: '',
    tools: [] as string[],
    mode: 'sequential' as 'sequential' | 'parallel' | 'reactive'
  });
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);

  const loadData = async () => {
    try {
      setLoading(true);

      // Load orchestrator status
      const statusResponse = await apiService.getOrchestratorStatus();
      if (statusResponse.success) {
        setOrchestratorStatus(statusResponse.data);
      }

      // Load active plans
      const plansResponse = await apiService.getActivePlans();
      if (plansResponse.success) {
        setActivePlans(plansResponse.data);
      }

      // Load completed plans
      const completedResponse = await apiService.getCompletedPlans();
      if (completedResponse.success) {
        setCompletedPlans(completedResponse.data.slice(0, 10)); // Last 10
      }
    } catch (error) {
      console.error('Failed to load orchestrator data:', error);
    } finally {
      setLoading(false);
    }
  };

  const createPlan = async () => {
    if (!newPlan.goal.trim() || newPlan.tools.length === 0) return;

    setExecuting(true);
    try {
      const response = await apiService.createOrchestrationPlan({
        goal: newPlan.goal,
        tools_needed: newPlan.tools,
        mode: newPlan.mode
      });

      if (response.success) {
        await loadData();
        setNewPlan({
          goal: '',
          tools: [],
          mode: 'sequential'
        });
      }
    } catch (error) {
      console.error('Failed to create plan:', error);
    } finally {
      setExecuting(false);
    }
  };

  const executePlan = async (planId: string) => {
    try {
      await apiService.executePlan(planId);
      await loadData();
    } catch (error) {
      console.error('Failed to execute plan:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'running':
      case 'executing':
        return <RefreshCw className="h-4 w-4 text-blue-400 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-400" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'failed':
        return 'destructive';
      case 'running':
      case 'executing':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading Tool Orchestrator...</span>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Tool Orchestrator</h1>
          <p className="text-neutral-400">
            Coordinate multiple tools for complex operations
          </p>
        </div>
        <Button onClick={loadData} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Plans</CardTitle>
            <Activity className="h-4 w-4 text-blue-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{orchestratorStatus?.active_plans || 0}</div>
            <p className="text-xs text-neutral-400">Currently executing</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available Tools</CardTitle>
            <Layers className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{orchestratorStatus?.available_tools || 0}</div>
            <p className="text-xs text-neutral-400">
              {orchestratorStatus?.rust_tools_available ? 'Rust-powered' : 'Python fallback'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {orchestratorStatus?.metrics.success_rate ?
                `${(orchestratorStatus.metrics.success_rate * 100).toFixed(1)}%` :
                '0%'
              }
            </div>
            <p className="text-xs text-neutral-400">
              {orchestratorStatus?.metrics.plans_executed || 0} plans executed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Time</CardTitle>
            <Clock className="h-4 w-4 text-yellow-400" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {orchestratorStatus?.metrics.avg_execution_time ?
                `${orchestratorStatus.metrics.avg_execution_time.toFixed(1)}s` :
                '0s'
              }
            </div>
            <p className="text-xs text-neutral-400">
              {orchestratorStatus?.metrics.tools_called || 0} tools called
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="create" className="space-y-4">
        <TabsList>
          <TabsTrigger value="create">Create Plan</TabsTrigger>
          <TabsTrigger value="active">Active Plans</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="tools">Available Tools</TabsTrigger>
        </TabsList>

        {/* Create Plan Tab */}
        <TabsContent value="create" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Create Orchestration Plan</CardTitle>
              <CardDescription>
                Define a multi-tool operation to execute sequentially or in parallel
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium">Goal</label>
                <textarea
                  className="w-full mt-1 p-3 border rounded-md bg-neutral-800 border-neutral-600 h-20"
                  placeholder="What do you want to accomplish?"
                  value={newPlan.goal}
                  onChange={(e) => setNewPlan(prev => ({ ...prev, goal: e.target.value }))}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Execution Mode</label>
                  <select
                    className="w-full mt-1 p-2 border rounded-md bg-neutral-800 border-neutral-600"
                    value={newPlan.mode}
                    onChange={(e) => setNewPlan(prev => ({
                      ...prev,
                      mode: e.target.value as 'sequential' | 'parallel' | 'reactive'
                    }))}
                  >
                    <option value="sequential">Sequential</option>
                    <option value="parallel">Parallel</option>
                    <option value="reactive">Reactive</option>
                  </select>
                </div>
                <div>
                  <label className="text-sm font-medium">Tools Needed</label>
                  <div className="mt-1 space-y-2">
                    <div className="flex flex-wrap gap-2">
                      {newPlan.tools.map((tool, index) => (
                        <Badge
                          key={index}
                          variant="secondary"
                          className="cursor-pointer"
                          onClick={() => setNewPlan(prev => ({
                            ...prev,
                            tools: prev.tools.filter((_, i) => i !== index)
                          }))}
                        >
                          {tool} ×
                        </Badge>
                      ))}
                    </div>
                    <select
                      className="w-full p-2 border rounded-md bg-neutral-800 border-neutral-600"
                      onChange={(e) => {
                        if (e.target.value && !newPlan.tools.includes(e.target.value)) {
                          setNewPlan(prev => ({
                            ...prev,
                            tools: [...prev.tools, e.target.value]
                          }));
                        }
                        e.target.value = '';
                      }}
                    >
                      <option value="">Select tools...</option>
                      {orchestratorStatus?.tool_registry.map(tool => (
                        <option key={tool} value={tool}>{tool}</option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              <Button
                onClick={createPlan}
                disabled={executing || !newPlan.goal.trim() || newPlan.tools.length === 0}
                className="w-full"
              >
                {executing ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Creating Plan...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Create & Execute Plan
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Active Plans Tab */}
        <TabsContent value="active" className="space-y-4">
          {activePlans.length === 0 ? (
            <Card>
              <CardContent className="text-center py-8 text-neutral-400">
                No active plans
              </CardContent>
            </Card>
          ) : (
            activePlans.map(plan => (
              <Card key={plan.plan_id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{plan.goal}</CardTitle>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getStatusBadgeVariant(plan.status)}>
                        {getStatusIcon(plan.status)}
                        <span className="ml-1">{plan.status}</span>
                      </Badge>
                      <Badge variant="outline">{plan.mode}</Badge>
                    </div>
                  </div>
                  <CardDescription>
                    {plan.tasks.length} tasks • {plan.tasks.filter(t => t.status === 'completed').length} completed
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {plan.tasks.map(task => (
                      <div key={task.id} className="flex items-center justify-between p-3 border rounded-md">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(task.status)}
                          <span className="font-medium">{task.tool_name}</span>
                          {task.execution_time && (
                            <span className="text-xs text-neutral-400">
                              ({task.execution_time.toFixed(2)}s)
                            </span>
                          )}
                        </div>
                        <Badge variant={getStatusBadgeVariant(task.status)}>
                          {task.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))
          )}
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-4">
          <ScrollArea className="h-96">
            {completedPlans.length === 0 ? (
              <div className="text-center py-8 text-neutral-400">
                No completed plans
              </div>
            ) : (
              <div className="space-y-4">
                {completedPlans.map(plan => (
                  <Card key={plan.plan_id}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm">{plan.goal}</CardTitle>
                        <div className="flex items-center space-x-2">
                          <Badge variant={getStatusBadgeVariant(plan.status)}>
                            {getStatusIcon(plan.status)}
                            <span className="ml-1">{plan.status}</span>
                          </Badge>
                          {plan.total_execution_time && (
                            <span className="text-xs text-neutral-400">
                              {plan.total_execution_time.toFixed(2)}s
                            </span>
                          )}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="text-sm text-neutral-400">
                        {plan.tasks.length} tasks • {plan.tasks.filter(t => t.status === 'completed').length} completed
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </ScrollArea>
        </TabsContent>

        {/* Available Tools Tab */}
        <TabsContent value="tools" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Available Tools</CardTitle>
              <CardDescription>
                Tools registered with the orchestrator
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                {orchestratorStatus?.tool_registry.map(tool => (
                  <Badge key={tool} variant="outline" className="justify-center p-2">
                    {tool}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
