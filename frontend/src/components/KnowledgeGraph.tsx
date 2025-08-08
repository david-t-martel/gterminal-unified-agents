import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Share2,
  RefreshCw,
  Search,
  Filter,
  Maximize2,
  Minimize2,
  Download,
  Upload,
  Settings,
  Eye,
  EyeOff,
  GitBranch,
  Database,
  Code,
  File,
  Package,
  Zap,
  Target,
  Info
} from 'lucide-react';
import {
  KnowledgeGraph as KnowledgeGraphType,
  KnowledgeNode,
  KnowledgeEdge,
  LanguageType,
  ProjectContext
} from '@/types/api';
import { apiService } from '@/services/api';

interface GraphVisualizationProps {
  graph: KnowledgeGraphType;
  selectedNode: KnowledgeNode | null;
  onNodeSelect: (node: KnowledgeNode | null) => void;
  visibleNodeTypes: Set<string>;
  visibleEdgeTypes: Set<string>;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  graph,
  selectedNode,
  onNodeSelect,
  visibleNodeTypes,
  visibleEdgeTypes
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Filter nodes and edges based on visibility settings
  const visibleNodes = graph.nodes.filter(node => visibleNodeTypes.has(node.type));
  const visibleEdges = graph.edges.filter(edge =>
    visibleEdgeTypes.has(edge.type) &&
    visibleNodes.some(n => n.id === edge.source) &&
    visibleNodes.some(n => n.id === edge.target)
  );

  // Simple force-directed layout calculation
  const calculateNodePositions = useCallback(() => {
    const nodePositions = new Map<string, { x: number; y: number }>();
    const width = 800;
    const height = 600;

    // Initialize positions randomly
    visibleNodes.forEach((node, index) => {
      const angle = (index / visibleNodes.length) * 2 * Math.PI;
      const radius = Math.min(width, height) * 0.3;
      nodePositions.set(node.id, {
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius
      });
    });

    return nodePositions;
  }, [visibleNodes]);

  const nodePositions = calculateNodePositions();

  const getNodeColor = (type: string) => {
    switch (type) {
      case 'file': return '#3b82f6';
      case 'function': return '#10b981';
      case 'class': return '#f59e0b';
      case 'module': return '#8b5cf6';
      case 'concept': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getEdgeColor = (type: string) => {
    switch (type) {
      case 'imports': return '#3b82f6';
      case 'calls': return '#10b981';
      case 'inherits': return '#f59e0b';
      case 'implements': return '#8b5cf6';
      case 'references': return '#ef4444';
      case 'contains': return '#6b7280';
      default: return '#4b5563';
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - panX, y: e.clientY - panY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPanX(e.clientX - dragStart.x);
      setPanY(e.clientY - dragStart.y);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleZoom = (delta: number) => {
    setZoom(prev => Math.max(0.1, Math.min(3, prev + delta)));
  };

  const handleNodeClick = (node: KnowledgeNode) => {
    onNodeSelect(selectedNode?.id === node.id ? null : node);
  };

  return (
    <div className="relative w-full h-96 border border-neutral-700 rounded-lg bg-neutral-900 overflow-hidden">
      {/* Controls */}
      <div className="absolute top-2 right-2 z-10 flex gap-1">
        <Button
          size="sm"
          variant="outline"
          onClick={() => handleZoom(0.1)}
          className="h-8 w-8 p-0"
        >
          +
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => handleZoom(-0.1)}
          className="h-8 w-8 p-0"
        >
          -
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            setZoom(1);
            setPanX(0);
            setPanY(0);
          }}
          className="h-8 w-8 p-0"
        >
          <Target className="w-3 h-3" />
        </Button>
      </div>

      {/* SVG Graph */}
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        className="cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={(e) => {
          e.preventDefault();
          handleZoom(e.deltaY > 0 ? -0.1 : 0.1);
        }}
      >
        <g transform={`translate(${panX}, ${panY}) scale(${zoom})`}>
          {/* Render edges */}
          {visibleEdges.map((edge) => {
            const sourcePos = nodePositions.get(edge.source);
            const targetPos = nodePositions.get(edge.target);

            if (!sourcePos || !targetPos) return null;

            return (
              <g key={edge.id}>
                <line
                  x1={sourcePos.x}
                  y1={sourcePos.y}
                  x2={targetPos.x}
                  y2={targetPos.y}
                  stroke={getEdgeColor(edge.type)}
                  strokeWidth={Math.max(1, edge.weight * 3)}
                  strokeOpacity={0.6}
                />
                {/* Edge label */}
                <text
                  x={(sourcePos.x + targetPos.x) / 2}
                  y={(sourcePos.y + targetPos.y) / 2}
                  fill="#9ca3af"
                  fontSize="10"
                  textAnchor="middle"
                  className="pointer-events-none"
                >
                  {edge.type}
                </text>
              </g>
            );
          })}

          {/* Render nodes */}
          {visibleNodes.map((node) => {
            const pos = nodePositions.get(node.id);
            if (!pos) return null;

            const isSelected = selectedNode?.id === node.id;
            const radius = 8 + (node.metadata.confidence * 10);

            return (
              <g key={node.id}>
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={radius}
                  fill={getNodeColor(node.type)}
                  stroke={isSelected ? '#ffffff' : 'none'}
                  strokeWidth={isSelected ? 2 : 0}
                  className="cursor-pointer hover:opacity-80"
                  onClick={() => handleNodeClick(node)}
                />
                <text
                  x={pos.x}
                  y={pos.y + radius + 12}
                  fill="#e5e7eb"
                  fontSize="11"
                  textAnchor="middle"
                  className="pointer-events-none select-none"
                >
                  {node.label.length > 15 ? `${node.label.slice(0, 12)}...` : node.label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      {/* Info overlay */}
      <div className="absolute bottom-2 left-2 text-xs text-neutral-400">
        {visibleNodes.length} nodes, {visibleEdges.length} edges
      </div>
    </div>
  );
};

interface NodeDetailsProps {
  node: KnowledgeNode | null;
  graph: KnowledgeGraphType;
}

const NodeDetails: React.FC<NodeDetailsProps> = ({ node, graph }) => {
  if (!node) {
    return (
      <Card className="p-4 border-neutral-700 bg-neutral-800/50 h-96 flex items-center justify-center">
        <div className="text-center">
          <Eye className="w-8 h-8 mx-auto text-neutral-400 mb-2" />
          <p className="text-neutral-400">Select a node to view details</p>
        </div>
      </Card>
    );
  }

  const connectedEdges = graph.edges.filter(
    edge => edge.source === node.id || edge.target === node.id
  );

  const incomingEdges = connectedEdges.filter(edge => edge.target === node.id);
  const outgoingEdges = connectedEdges.filter(edge => edge.source === node.id);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'file': return <File className="w-4 h-4" />;
      case 'function': return <Zap className="w-4 h-4" />;
      case 'class': return <Package className="w-4 h-4" />;
      case 'module': return <Database className="w-4 h-4" />;
      case 'concept': return <GitBranch className="w-4 h-4" />;
      default: return <Code className="w-4 h-4" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'file': return 'text-blue-400';
      case 'function': return 'text-green-400';
      case 'class': return 'text-yellow-400';
      case 'module': return 'text-purple-400';
      case 'concept': return 'text-red-400';
      default: return 'text-neutral-400';
    }
  };

  return (
    <Card className="p-4 border-neutral-700 bg-neutral-800/50">
      <div className="space-y-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <div className={getTypeColor(node.type)}>
              {getTypeIcon(node.type)}
            </div>
            <h3 className="font-semibold text-neutral-100">{node.label}</h3>
            <Badge variant="outline" className="text-xs">
              {node.type}
            </Badge>
          </div>

          <div className="text-sm text-neutral-400">
            ID: <code className="text-neutral-300">{node.id}</code>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-neutral-300 mb-2">Metadata</h4>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-neutral-400">Source:</span>
              <span className="text-neutral-300">{node.metadata.source}</span>
            </div>
            {node.metadata.language && (
              <div className="flex justify-between">
                <span className="text-neutral-400">Language:</span>
                <Badge className="text-xs">{node.metadata.language}</Badge>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-neutral-400">Confidence:</span>
              <span className="text-neutral-300">
                {(node.metadata.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {Object.keys(node.properties).length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-neutral-300 mb-2">Properties</h4>
            <ScrollArea className="h-24">
              <div className="space-y-1 text-sm">
                {Object.entries(node.properties).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-neutral-400">{key}:</span>
                    <span className="text-neutral-300 max-w-32 truncate">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

        <div>
          <h4 className="text-sm font-medium text-neutral-300 mb-2">Connections</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <h5 className="text-xs font-medium text-neutral-400 mb-1">
                Incoming ({incomingEdges.length})
              </h5>
              <ScrollArea className="h-16">
                <div className="space-y-1">
                  {incomingEdges.map((edge, index) => (
                    <div key={index} className="text-xs text-neutral-300">
                      {edge.type} from {graph.nodes.find(n => n.id === edge.source)?.label || edge.source}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>

            <div>
              <h5 className="text-xs font-medium text-neutral-400 mb-1">
                Outgoing ({outgoingEdges.length})
              </h5>
              <ScrollArea className="h-16">
                <div className="space-y-1">
                  {outgoingEdges.map((edge, index) => (
                    <div key={index} className="text-xs text-neutral-300">
                      {edge.type} to {graph.nodes.find(n => n.id === edge.target)?.label || edge.target}
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export const KnowledgeGraph: React.FC = () => {
  const [graph, setGraph] = useState<KnowledgeGraphType | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<KnowledgeNode | null>(null);
  const [activeTab, setActiveTab] = useState('graph');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedContext, setSelectedContext] = useState<string>('');
  const [contexts, setContexts] = useState<ProjectContext[]>([]);

  // Visibility controls
  const [visibleNodeTypes, setVisibleNodeTypes] = useState<Set<string>>(
    new Set(['file', 'function', 'class', 'module', 'concept'])
  );
  const [visibleEdgeTypes, setVisibleEdgeTypes] = useState<Set<string>>(
    new Set(['imports', 'calls', 'inherits', 'implements', 'references', 'contains'])
  );

  const loadContexts = useCallback(async () => {
    try {
      const response = await apiService.listContexts();
      if (response.success) {
        setContexts(response.data);
      }
    } catch (err) {
      console.error('Failed to load contexts:', err);
    }
  }, []);

  const loadGraph = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await apiService.getKnowledgeGraph(selectedContext || undefined);
      if (response.success) {
        setGraph(response.data);
      } else {
        setError(response.error?.message || 'Failed to load knowledge graph');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load knowledge graph');
    } finally {
      setLoading(false);
    }
  }, [selectedContext]);

  const toggleNodeType = useCallback((type: string) => {
    setVisibleNodeTypes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(type)) {
        newSet.delete(type);
      } else {
        newSet.add(type);
      }
      return newSet;
    });
  }, []);

  const toggleEdgeType = useCallback((type: string) => {
    setVisibleEdgeTypes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(type)) {
        newSet.delete(type);
      } else {
        newSet.add(type);
      }
      return newSet;
    });
  }, []);

  const exportGraph = useCallback(() => {
    if (!graph) return;

    const blob = new Blob([JSON.stringify(graph, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `knowledge-graph-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [graph]);

  const filteredNodes = graph ? graph.nodes.filter(node =>
    node.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    node.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
    node.metadata.source.toLowerCase().includes(searchQuery.toLowerCase())
  ) : [];

  useEffect(() => {
    loadContexts();
  }, [loadContexts]);

  useEffect(() => {
    loadGraph();
  }, [loadGraph]);

  const nodeTypes = graph ? [...new Set(graph.nodes.map(n => n.type))] : [];
  const edgeTypes = graph ? [...new Set(graph.edges.map(e => e.type))] : [];

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Knowledge Graph</h1>
          <p className="text-neutral-400">Visualize relationships in your codebase and knowledge</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadGraph} variant="outline" size="sm" disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          {graph && (
            <Button onClick={exportGraph} variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          )}
        </div>
      </div>

      <Card className="p-4 border-neutral-700 bg-neutral-800/50">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="text-sm font-medium text-neutral-300 block mb-2">
              Context
            </label>
            <Select value={selectedContext} onValueChange={setSelectedContext}>
              <SelectTrigger className="bg-neutral-700 border-neutral-600">
                <SelectValue placeholder="Select context (or leave empty for global)" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">Global knowledge graph</SelectItem>
                {contexts.map((context) => (
                  <SelectItem key={context.id} value={context.id}>
                    {context.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex-1">
            <label className="text-sm font-medium text-neutral-300 block mb-2">
              Search Nodes
            </label>
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search nodes by label, type, or source..."
              className="bg-neutral-700 border-neutral-600"
            />
          </div>
        </div>
      </Card>

      {error && (
        <Card className="p-4 border-red-500/30 bg-red-500/10">
          <div className="flex items-center gap-2 text-red-300">
            <Info className="w-4 h-4" />
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
          <TabsTrigger value="graph">Graph</TabsTrigger>
          <TabsTrigger value="nodes">Nodes</TabsTrigger>
          <TabsTrigger value="controls">Controls</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
        </TabsList>

        <TabsContent value="graph" className="mt-6">
          {graph ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <GraphVisualization
                  graph={graph}
                  selectedNode={selectedNode}
                  onNodeSelect={setSelectedNode}
                  visibleNodeTypes={visibleNodeTypes}
                  visibleEdgeTypes={visibleEdgeTypes}
                />
              </div>
              <NodeDetails node={selectedNode} graph={graph} />
            </div>
          ) : loading ? (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <RefreshCw className="w-8 h-8 mx-auto text-neutral-400 mb-4 animate-spin" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                Loading Knowledge Graph
              </h3>
              <p className="text-neutral-400">
                Processing relationships and building visualization...
              </p>
            </Card>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Share2 className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Knowledge Graph Available
              </h3>
              <p className="text-neutral-400">
                Select a context or ensure your project has been analyzed
              </p>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="nodes" className="mt-6">
          {graph ? (
            <div className="grid gap-4">
              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <h3 className="font-semibold text-neutral-100 mb-3">
                  Nodes {searchQuery && `(filtered: ${filteredNodes.length}/${graph.nodes.length})`}
                </h3>

                <ScrollArea className="h-96">
                  <div className="space-y-2">
                    {filteredNodes.map((node) => (
                      <div
                        key={node.id}
                        className={`p-3 rounded-md cursor-pointer transition-colors ${
                          selectedNode?.id === node.id
                            ? 'bg-blue-500/20 border border-blue-500/30'
                            : 'bg-neutral-700/50 hover:bg-neutral-700'
                        }`}
                        onClick={() => setSelectedNode(node)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              {node.type}
                            </Badge>
                            <span className="font-medium text-neutral-100">{node.label}</span>
                          </div>
                          <span className="text-xs text-neutral-400">
                            {(node.metadata.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <p className="text-sm text-neutral-400 mt-1">
                          {node.metadata.source}
                        </p>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </Card>
            </div>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Database className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Nodes Available
              </h3>
              <p className="text-neutral-400">
                Load a knowledge graph to view nodes
              </p>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="controls" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <h3 className="font-semibold text-neutral-100 mb-3">Node Types</h3>
              <div className="space-y-2">
                {nodeTypes.map((type) => (
                  <label key={type} className="flex items-center gap-2 text-sm text-neutral-300">
                    <input
                      type="checkbox"
                      checked={visibleNodeTypes.has(type)}
                      onChange={() => toggleNodeType(type)}
                      className="rounded"
                    />
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                    <Badge variant="outline" className="text-xs ml-auto">
                      {graph?.nodes.filter(n => n.type === type).length || 0}
                    </Badge>
                  </label>
                ))}
              </div>

              <div className="flex gap-2 mt-4">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setVisibleNodeTypes(new Set(nodeTypes))}
                >
                  Show All
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setVisibleNodeTypes(new Set())}
                >
                  Hide All
                </Button>
              </div>
            </Card>

            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <h3 className="font-semibold text-neutral-100 mb-3">Edge Types</h3>
              <div className="space-y-2">
                {edgeTypes.map((type) => (
                  <label key={type} className="flex items-center gap-2 text-sm text-neutral-300">
                    <input
                      type="checkbox"
                      checked={visibleEdgeTypes.has(type)}
                      onChange={() => toggleEdgeType(type)}
                      className="rounded"
                    />
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                    <Badge variant="outline" className="text-xs ml-auto">
                      {graph?.edges.filter(e => e.type === type).length || 0}
                    </Badge>
                  </label>
                ))}
              </div>

              <div className="flex gap-2 mt-4">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setVisibleEdgeTypes(new Set(edgeTypes))}
                >
                  Show All
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setVisibleEdgeTypes(new Set())}
                >
                  Hide All
                </Button>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="stats" className="mt-6">
          {graph ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="w-5 h-5 text-blue-400" />
                  <h3 className="font-semibold text-neutral-100">Total Nodes</h3>
                </div>
                <p className="text-2xl font-bold text-neutral-100">{graph.metadata.nodeCount}</p>
                <p className="text-sm text-neutral-400">
                  {visibleNodeTypes.size}/{nodeTypes.length} types visible
                </p>
              </Card>

              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <div className="flex items-center gap-2 mb-2">
                  <GitBranch className="w-5 h-5 text-green-400" />
                  <h3 className="font-semibold text-neutral-100">Total Edges</h3>
                </div>
                <p className="text-2xl font-bold text-neutral-100">{graph.metadata.edgeCount}</p>
                <p className="text-sm text-neutral-400">
                  {visibleEdgeTypes.size}/{edgeTypes.length} types visible
                </p>
              </Card>

              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-5 h-5 text-purple-400" />
                  <h3 className="font-semibold text-neutral-100">Avg Confidence</h3>
                </div>
                <p className="text-2xl font-bold text-neutral-100">
                  {graph.nodes.length > 0
                    ? ((graph.nodes.reduce((sum, n) => sum + n.metadata.confidence, 0) / graph.nodes.length) * 100).toFixed(1)
                    : 0
                  }%
                </p>
                <p className="text-sm text-neutral-400">node confidence</p>
              </Card>

              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <div className="flex items-center gap-2 mb-2">
                  <RefreshCw className="w-5 h-5 text-yellow-400" />
                  <h3 className="font-semibold text-neutral-100">Last Updated</h3>
                </div>
                <p className="text-sm font-bold text-neutral-100">
                  {new Date(graph.metadata.lastUpdated).toLocaleDateString()}
                </p>
                <p className="text-sm text-neutral-400">
                  {new Date(graph.metadata.lastUpdated).toLocaleTimeString()}
                </p>
              </Card>
            </div>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Target className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Statistics Available
              </h3>
              <p className="text-neutral-400">
                Load a knowledge graph to view statistics
              </p>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};
