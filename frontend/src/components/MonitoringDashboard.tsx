import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { AlertTriangle, Activity, Zap, Users, Server, TrendingUp, RefreshCw } from 'lucide-react';

// Types for monitoring data
interface SystemHealth {
  timestamp: string;
  overall_status: string;
  components: {
    apm: {
      status: string;
      operations: number;
      error_rate: number;
      response_time_p95: number;
    };
    rum: {
      active_users: number;
      bounce_rate: number;
      avg_response_time: number;
      errors_last_hour: number;
    };
    ai_metrics: {
      status: string;
      total_operations: number;
      error_rate: number;
      cost_per_hour: number;
    };
    slo: {
      compliant_slos: number;
      violated_slos: number;
      total_slos: number;
      compliance_rate: number;
    };
    incidents: {
      active_count: number;
      resolved_today: number;
      mttr_minutes: number;
    };
  };
}

interface RealTimeMetrics {
  timestamp: string;
  performance: {
    response_times: {
      avg_ms: number;
      p95_ms: number;
      p99_ms: number;
    };
    error_rates: {
      total_rate: number;
      ai_error_rate: number;
      user_errors_last_hour: number;
    };
    throughput: {
      requests_per_second: number;
      ai_operations_per_hour: number;
      active_users: number;
    };
  };
  resources: {
    memory_usage_mb: number;
    cpu_utilization: number;
    ai_cost_per_hour: number;
    cache_hit_rate: number;
  };
  incidents: {
    active_count: number;
    resolved_today: number;
    mttr_minutes: number;
  };
  slo_compliance: {
    compliant_count: number;
    violated_count: number;
    compliance_rate: number;
  };
}

interface Incident {
  id: string;
  title: string;
  severity: string;
  status: string;
  created_at: string;
  source: string;
}

interface PerformanceInsight {
  category: string;
  recommendation: string;
  priority: string;
}

export function MonitoringDashboard() {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [realTimeMetrics, setRealTimeMetrics] = useState<RealTimeMetrics | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [insights, setInsights] = useState<PerformanceInsight[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // API base URL
  const API_BASE = '/api/v1/monitoring';

  // Fetch system health data
  const fetchSystemHealth = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (response.ok) {
        const data = await response.json();
        setSystemHealth(data);
      }
    } catch (error) {
      console.error('Error fetching system health:', error);
    }
  }, [API_BASE]);

  // Fetch real-time metrics
  const fetchRealTimeMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics/realtime`);
      if (response.ok) {
        const data = await response.json();
        setRealTimeMetrics(data);
      }
    } catch (error) {
      console.error('Error fetching real-time metrics:', error);
    }
  }, [API_BASE]);

  // Fetch incidents
  const fetchIncidents = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/incidents`);
      if (response.ok) {
        const data = await response.json();
        setIncidents(data.active_incidents || []);
      }
    } catch (error) {
      console.error('Error fetching incidents:', error);
    }
  }, [API_BASE]);

  // Fetch performance insights
  const fetchInsights = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/insights`);
      if (response.ok) {
        const data = await response.json();
        setInsights(data.aggregated_recommendations || []);
      }
    } catch (error) {
      console.error('Error fetching insights:', error);
    }
  }, [API_BASE]);

  // Fetch all data
  const fetchAllData = useCallback(async () => {
    setIsLoading(true);
    await Promise.all([
      fetchSystemHealth(),
      fetchRealTimeMetrics(),
      fetchIncidents(),
      fetchInsights(),
    ]);
    setLastUpdate(new Date());
    setIsLoading(false);
  }, [fetchSystemHealth, fetchRealTimeMetrics, fetchIncidents, fetchInsights]);

  // Initial load
  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  // Auto-refresh
  useEffect(() => {
    if (!isAutoRefresh) return;

    const interval = setInterval(fetchAllData, 30000); // 30 seconds
    return () => clearInterval(interval);
  }, [isAutoRefresh, fetchAllData]);

  // Status badge helper
  const getStatusBadge = (status: string) => {
    const variants: Record<string, 'default' | 'destructive' | 'secondary' | 'outline'> = {
      healthy: 'default',
      operational: 'default',
      warning: 'secondary',
      degraded: 'secondary',
      critical: 'destructive',
      unhealthy: 'destructive',
    };

    return (
      <Badge variant={variants[status] || 'outline'} className="capitalize">
        {status}
      </Badge>
    );
  };

  // Format number helper
  const formatNumber = (num: number, decimals = 0) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toFixed(decimals);
  };

  if (isLoading && !systemHealth) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
          <span className="ml-2 text-lg">Loading monitoring data...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Monitoring</h1>
          <p className="text-muted-foreground mt-1">
            Real-time performance and health monitoring
            {lastUpdate && (
              <span className="ml-2">
                • Last updated: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAutoRefresh(!isAutoRefresh)}
          >
            <Activity className={`h-4 w-4 mr-2 ${isAutoRefresh ? 'animate-pulse' : ''}`} />
            Auto-refresh {isAutoRefresh ? 'ON' : 'OFF'}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchAllData}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Status Overview */}
      {systemHealth && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Server className="h-4 w-4 mr-2" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mb-1">
                {getStatusBadge(systemHealth.overall_status)}
              </div>
              <p className="text-xs text-muted-foreground">
                Overall system health
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Users className="h-4 w-4 mr-2" />
                Active Users
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mb-1">
                {formatNumber(systemHealth.components.rum.active_users)}
              </div>
              <p className="text-xs text-muted-foreground">
                Current active sessions
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <Zap className="h-4 w-4 mr-2" />
                AI Operations
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mb-1">
                {formatNumber(systemHealth.components.ai_metrics.total_operations)}
              </div>
              <p className="text-xs text-muted-foreground">
                Total AI operations
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium flex items-center">
                <AlertTriangle className="h-4 w-4 mr-2" />
                Active Incidents
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold mb-1">
                {systemHealth.components.incidents.active_count}
              </div>
              <p className="text-xs text-muted-foreground">
                Require attention
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Detailed Monitoring Tabs */}
      <Tabs defaultValue="performance" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="incidents">Incidents</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="sla">SLA Status</TabsTrigger>
        </TabsList>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          {realTimeMetrics && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Response Times */}
              <Card>
                <CardHeader>
                  <CardTitle>Response Times</CardTitle>
                  <CardDescription>API response performance metrics</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Average</span>
                      <span className="font-mono">{realTimeMetrics.performance.response_times.avg_ms.toFixed(0)}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span>95th Percentile</span>
                      <span className="font-mono">{realTimeMetrics.performance.response_times.p95_ms.toFixed(0)}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span>99th Percentile</span>
                      <span className="font-mono">{realTimeMetrics.performance.response_times.p99_ms.toFixed(0)}ms</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Error Rates */}
              <Card>
                <CardHeader>
                  <CardTitle>Error Rates</CardTitle>
                  <CardDescription>System error monitoring</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Total Error Rate</span>
                      <span className="font-mono">{realTimeMetrics.performance.error_rates.total_rate.toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>AI Error Rate</span>
                      <span className="font-mono">{realTimeMetrics.performance.error_rates.ai_error_rate.toFixed(2)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>User Errors (1h)</span>
                      <span className="font-mono">{realTimeMetrics.performance.error_rates.user_errors_last_hour}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Resource Usage */}
              <Card>
                <CardHeader>
                  <CardTitle>Resource Usage</CardTitle>
                  <CardDescription>System resource consumption</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>Memory Usage</span>
                        <span className="font-mono">{formatNumber(realTimeMetrics.resources.memory_usage_mb, 1)}MB</span>
                      </div>
                      <Progress value={(realTimeMetrics.resources.memory_usage_mb / 4096) * 100} />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>CPU Utilization</span>
                        <span className="font-mono">{realTimeMetrics.resources.cpu_utilization.toFixed(1)}%</span>
                      </div>
                      <Progress value={realTimeMetrics.resources.cpu_utilization} />
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span>Cache Hit Rate</span>
                        <span className="font-mono">{realTimeMetrics.resources.cache_hit_rate.toFixed(1)}%</span>
                      </div>
                      <Progress value={realTimeMetrics.resources.cache_hit_rate} />
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Throughput */}
              <Card>
                <CardHeader>
                  <CardTitle>Throughput</CardTitle>
                  <CardDescription>Request and operation volumes</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Requests/Second</span>
                      <span className="font-mono">{realTimeMetrics.performance.throughput.requests_per_second.toFixed(1)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>AI Ops/Hour</span>
                      <span className="font-mono">{formatNumber(realTimeMetrics.performance.throughput.ai_operations_per_hour)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>AI Cost/Hour</span>
                      <span className="font-mono">${realTimeMetrics.resources.ai_cost_per_hour.toFixed(2)}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Incidents Tab */}
        <TabsContent value="incidents" className="space-y-4">
          {incidents.length > 0 ? (
            <div className="space-y-4">
              {incidents.map((incident) => (
                <Card key={incident.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{incident.title}</CardTitle>
                      <div className="flex items-center gap-2">
                        {getStatusBadge(incident.severity)}
                        {getStatusBadge(incident.status)}
                      </div>
                    </div>
                    <CardDescription>
                      Created: {new Date(incident.created_at).toLocaleString()} • Source: {incident.source}
                    </CardDescription>
                  </CardHeader>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="pt-6">
                <div className="text-center text-muted-foreground">
                  <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No active incidents</p>
                  <p className="text-sm">All systems operating normally</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-4">
          {insights.length > 0 ? (
            <div className="space-y-4">
              {insights.map((insight, index) => (
                <Card key={index}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base capitalize">{insight.category}</CardTitle>
                      <Badge variant={insight.priority === 'high' ? 'destructive' : 'secondary'}>
                        {insight.priority} priority
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm">{insight.recommendation}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="pt-6">
                <div className="text-center text-muted-foreground">
                  <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No performance insights available</p>
                  <p className="text-sm">System is performing optimally</p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* SLA Status Tab */}
        <TabsContent value="sla" className="space-y-4">
          {systemHealth && (
            <Card>
              <CardHeader>
                <CardTitle>SLA Compliance</CardTitle>
                <CardDescription>Service Level Agreement monitoring</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {systemHealth.components.slo.compliant_slos}
                    </div>
                    <div className="text-sm text-green-600 font-medium">Compliant SLOs</div>
                  </div>
                  <div className="text-center p-4 bg-red-50 dark:bg-red-950 rounded-lg">
                    <div className="text-2xl font-bold text-red-600">
                      {systemHealth.components.slo.violated_slos}
                    </div>
                    <div className="text-sm text-red-600 font-medium">Violated SLOs</div>
                  </div>
                  <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {systemHealth.components.slo.compliance_rate.toFixed(1)}%
                    </div>
                    <div className="text-sm text-blue-600 font-medium">Compliance Rate</div>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="flex justify-between mb-2">
                    <span>Overall Compliance</span>
                    <span className="font-mono">{systemHealth.components.slo.compliance_rate.toFixed(1)}%</span>
                  </div>
                  <Progress value={systemHealth.components.slo.compliance_rate} />
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
