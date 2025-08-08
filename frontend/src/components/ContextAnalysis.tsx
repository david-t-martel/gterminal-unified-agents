import React, { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Folder,
  File,
  Search,
  AlertTriangle,
  Info,
  CheckCircle,
  RefreshCw,
  Upload,
  Download,
  Trash2,
  FolderOpen,
  Code,
  Shield,
  Zap,
  Wrench
} from 'lucide-react';
import {
  ProjectContext,
  ProjectInsight,
  ContextAnalyzeRequest,
  ContextSearchRequest,
  FileNode,
  LanguageType
} from '@/types/api';
import { apiService } from '@/services/api';

interface FileTreeProps {
  nodes: FileNode[];
  onNodeClick: (node: FileNode) => void;
  expandedNodes: Set<string>;
  onToggleExpand: (path: string) => void;
}

const FileTree: React.FC<FileTreeProps> = ({
  nodes,
  onNodeClick,
  expandedNodes,
  onToggleExpand
}) => {
  const renderNode = (node: FileNode, depth = 0) => {
    const isExpanded = expandedNodes.has(node.path);
    const hasChildren = node.children && node.children.length > 0;

    return (
      <div key={node.path}>
        <div
          className={`flex items-center gap-2 p-2 hover:bg-neutral-700/50 cursor-pointer rounded-md transition-colors ${
            depth > 0 ? 'ml-' + (depth * 4) : ''
          }`}
          onClick={() => {
            if (node.type === 'directory' && hasChildren) {
              onToggleExpand(node.path);
            }
            onNodeClick(node);
          }}
        >
          {node.type === 'directory' ? (
            <>
              {hasChildren && (
                <div className="w-4 h-4 flex items-center justify-center">
                  <div
                    className={`w-2 h-2 border-l border-b border-neutral-400 transform transition-transform ${
                      isExpanded ? 'rotate-45' : '-rotate-45'
                    }`}
                  />
                </div>
              )}
              {isExpanded ? (
                <FolderOpen className="w-4 h-4 text-blue-400" />
              ) : (
                <Folder className="w-4 h-4 text-blue-400" />
              )}
            </>
          ) : (
            <File className="w-4 h-4 text-neutral-400 ml-6" />
          )}

          <span className="text-sm text-neutral-300 truncate">{node.name}</span>

          {node.size && (
            <span className="text-xs text-neutral-500 ml-auto">
              {formatFileSize(node.size)}
            </span>
          )}
        </div>

        {node.type === 'directory' && hasChildren && isExpanded && (
          <div className="ml-4">
            {node.children!.map(child => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-1">
      {nodes.map(node => renderNode(node))}
    </div>
  );
};

interface InsightCardProps {
  insight: ProjectInsight;
}

const InsightCard: React.FC<InsightCardProps> = ({ insight }) => {
  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'architecture': return <Code className="w-4 h-4" />;
      case 'security': return <Shield className="w-4 h-4" />;
      case 'performance': return <Zap className="w-4 h-4" />;
      case 'maintainability': return <Wrench className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'error': return 'bg-red-500/20 text-red-300 border-red-500/30';
      case 'warning': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      default: return 'bg-blue-500/20 text-blue-300 border-blue-500/30';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'architecture': return 'text-purple-400';
      case 'security': return 'text-red-400';
      case 'performance': return 'text-green-400';
      case 'maintainability': return 'text-blue-400';
      default: return 'text-neutral-400';
    }
  };

  return (
    <Card className="p-4 border-neutral-700 bg-neutral-800/50">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={getTypeColor(insight.type)}>
            {getInsightIcon(insight.type)}
          </div>
          <h3 className="font-semibold text-neutral-100">{insight.title}</h3>
        </div>
        <Badge className={`${getSeverityColor(insight.severity)} border`}>
          {insight.severity}
        </Badge>
      </div>

      <p className="text-neutral-300 text-sm mb-3">{insight.description}</p>

      {insight.location && (
        <div className="text-xs text-neutral-400 mb-2">
          <File className="w-3 h-3 inline mr-1" />
          {insight.location.file}
          {insight.location.line && `:${insight.location.line}`}
          {insight.location.column && `:${insight.location.column}`}
        </div>
      )}

      {insight.suggestion && (
        <div className="bg-neutral-700/50 p-3 rounded-md">
          <h4 className="text-xs font-semibold text-neutral-200 mb-1">Suggestion:</h4>
          <p className="text-xs text-neutral-300">{insight.suggestion}</p>
        </div>
      )}
    </Card>
  );
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

export const ContextAnalysis: React.FC = () => {
  const [contexts, setContexts] = useState<ProjectContext[]>([]);
  const [selectedContext, setSelectedContext] = useState<ProjectContext | null>(null);
  const [insights, setInsights] = useState<ProjectInsight[]>([]);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [activeTab, setActiveTab] = useState('structure');
  const [projectPath, setProjectPath] = useState('/home/david/agents/my-fullstack-agent');

  const loadContexts = useCallback(async () => {
    try {
      setLoading(true);
      const response = await apiService.listContexts();
      if (response.success) {
        setContexts(response.data);
        if (response.data.length > 0 && !selectedContext) {
          setSelectedContext(response.data[0]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load contexts');
    } finally {
      setLoading(false);
    }
  }, [selectedContext]);

  const analyzeProject = useCallback(async () => {
    if (!projectPath.trim()) return;

    try {
      setAnalyzing(true);
      setError(null);

      const request: ContextAnalyzeRequest = {
        projectPath: projectPath,
        analysisDepth: 3,
        excludePatterns: ['node_modules', '.git', '__pycache__', '.venv']
      };

      const response = await apiService.analyzeContext(request);
      if (response.success) {
        setSelectedContext(response.data.context);
        setInsights(response.data.insights);
        await loadContexts(); // Refresh the contexts list
      } else {
        setError(response.error?.message || 'Analysis failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  }, [projectPath, loadContexts]);

  const searchContexts = useCallback(async () => {
    if (!searchQuery.trim()) return;

    try {
      setLoading(true);
      const request: ContextSearchRequest = {
        query: searchQuery,
        contextId: selectedContext?.id
      };

      const response = await apiService.searchContext(request);
      if (response.success) {
        setSearchResults(response.data.results);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [searchQuery, selectedContext?.id]);

  const deleteContext = useCallback(async (contextId: string) => {
    try {
      const response = await apiService.deleteContext(contextId);
      if (response.success) {
        setContexts(prev => prev.filter(c => c.id !== contextId));
        if (selectedContext?.id === contextId) {
          setSelectedContext(null);
          setInsights([]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete context');
    }
  }, [selectedContext?.id]);

  const handleNodeClick = useCallback((node: FileNode) => {
    console.log('Selected node:', node);
  }, []);

  const handleToggleExpand = useCallback((path: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  }, []);

  useEffect(() => {
    loadContexts();
  }, [loadContexts]);

  const getLanguageColor = (language: LanguageType) => {
    switch (language) {
      case LanguageType.PYTHON: return 'bg-blue-500/20 text-blue-300';
      case LanguageType.TYPESCRIPT: return 'bg-blue-600/20 text-blue-400';
      case LanguageType.JAVASCRIPT: return 'bg-yellow-500/20 text-yellow-300';
      case LanguageType.RUST: return 'bg-orange-500/20 text-orange-300';
      case LanguageType.GO: return 'bg-cyan-500/20 text-cyan-300';
      case LanguageType.JAVA: return 'bg-red-500/20 text-red-300';
      default: return 'bg-neutral-500/20 text-neutral-300';
    }
  };

  const groupedInsights = insights.reduce((acc, insight) => {
    if (!acc[insight.type]) acc[insight.type] = [];
    acc[insight.type].push(insight);
    return acc;
  }, {} as Record<string, ProjectInsight[]>);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Context Analysis</h1>
          <p className="text-neutral-400">Analyze and explore project structure and insights</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadContexts} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <Card className="p-4 border-neutral-700 bg-neutral-800/50">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="text-sm font-medium text-neutral-300 block mb-2">
              Project Path
            </label>
            <Input
              value={projectPath}
              onChange={(e) => setProjectPath(e.target.value)}
              placeholder="/path/to/your/project"
              className="bg-neutral-700 border-neutral-600"
            />
          </div>
          <Button
            onClick={analyzeProject}
            disabled={analyzing || !projectPath.trim()}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {analyzing ? (
              <>
                <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="w-4 h-4 mr-2" />
                Analyze Project
              </>
            )}
          </Button>
        </div>
      </Card>

      {error && (
        <Card className="p-4 border-red-500/30 bg-red-500/10">
          <div className="flex items-center gap-2 text-red-300">
            <AlertTriangle className="w-4 h-4" />
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Contexts Sidebar */}
        <Card className="p-4 border-neutral-700 bg-neutral-800/50">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-neutral-100">Project Contexts</h2>
            <Badge variant="outline" className="border-neutral-600 text-neutral-300">
              {contexts.length}
            </Badge>
          </div>

          <ScrollArea className="h-64">
            <div className="space-y-2">
              {contexts.map((context) => (
                <div
                  key={context.id}
                  className={`p-3 rounded-md cursor-pointer transition-colors ${
                    selectedContext?.id === context.id
                      ? 'bg-blue-500/20 border border-blue-500/30'
                      : 'bg-neutral-700/50 hover:bg-neutral-700'
                  }`}
                  onClick={() => setSelectedContext(context)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-medium text-neutral-100 text-sm">{context.name}</h3>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteContext(context.id);
                      }}
                      className="h-6 w-6 p-0 text-neutral-400 hover:text-red-400"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                  <div className="flex items-center gap-2 text-xs">
                    <Badge className={`${getLanguageColor(context.language)} px-2 py-0`}>
                      {context.language}
                    </Badge>
                    <span className="text-neutral-400">
                      {context.structure.totalFiles} files
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        </Card>

        {/* Main Content */}
        <div className="lg:col-span-2">
          {selectedContext ? (
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-4 bg-neutral-800">
                <TabsTrigger value="structure">Structure</TabsTrigger>
                <TabsTrigger value="insights">Insights</TabsTrigger>
                <TabsTrigger value="dependencies">Dependencies</TabsTrigger>
                <TabsTrigger value="search">Search</TabsTrigger>
              </TabsList>

              <TabsContent value="structure" className="mt-4">
                <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-neutral-100">Project Structure</h3>
                    <div className="flex gap-4 text-sm text-neutral-400">
                      <span>{selectedContext.structure.totalFiles} files</span>
                      <span>{selectedContext.structure.totalDirectories} directories</span>
                    </div>
                  </div>

                  <ScrollArea className="h-96">
                    <FileTree
                      nodes={selectedContext.structure.files}
                      onNodeClick={handleNodeClick}
                      expandedNodes={expandedNodes}
                      onToggleExpand={handleToggleExpand}
                    />
                  </ScrollArea>
                </Card>
              </TabsContent>

              <TabsContent value="insights" className="mt-4">
                <div className="space-y-4">
                  {Object.keys(groupedInsights).length === 0 ? (
                    <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
                      <CheckCircle className="w-12 h-12 mx-auto text-green-400 mb-4" />
                      <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                        No Issues Found
                      </h3>
                      <p className="text-neutral-400">
                        Your project structure looks good!
                      </p>
                    </Card>
                  ) : (
                    Object.entries(groupedInsights).map(([type, typeInsights]) => (
                      <div key={type}>
                        <h3 className="font-semibold text-neutral-100 mb-3 capitalize">
                          {type} ({typeInsights.length})
                        </h3>
                        <div className="grid gap-4 md:grid-cols-2">
                          {typeInsights.map((insight, index) => (
                            <InsightCard key={index} insight={insight} />
                          ))}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </TabsContent>

              <TabsContent value="dependencies" className="mt-4">
                <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                  <h3 className="font-semibold text-neutral-100 mb-4">Dependencies</h3>

                  <div className="space-y-4">
                    {selectedContext.dependencies.map((dep, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-neutral-700/50 rounded-md">
                        <div>
                          <h4 className="font-medium text-neutral-100">{dep.name}</h4>
                          <div className="flex gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              {dep.version}
                            </Badge>
                            <Badge variant="outline" className="text-xs">
                              {dep.type}
                            </Badge>
                            <Badge className={`${getLanguageColor(dep.language)} text-xs`}>
                              {dep.language}
                            </Badge>
                          </div>
                        </div>

                        {dep.vulnerabilities && dep.vulnerabilities.length > 0 && (
                          <Badge className="bg-red-500/20 text-red-300 border-red-500/30">
                            {dep.vulnerabilities.length} vulnerabilities
                          </Badge>
                        )}
                      </div>
                    ))}
                  </div>
                </Card>
              </TabsContent>

              <TabsContent value="search" className="mt-4">
                <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                  <div className="flex gap-4 mb-4">
                    <Input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search project contexts..."
                      className="flex-1 bg-neutral-700 border-neutral-600"
                      onKeyPress={(e) => e.key === 'Enter' && searchContexts()}
                    />
                    <Button onClick={searchContexts} disabled={!searchQuery.trim()}>
                      <Search className="w-4 h-4 mr-2" />
                      Search
                    </Button>
                  </div>

                  <ScrollArea className="h-64">
                    <div className="space-y-2">
                      {searchResults.map((result, index) => (
                        <div key={index} className="p-3 bg-neutral-700/50 rounded-md">
                          <h4 className="font-medium text-neutral-100">{result.context.name}</h4>
                          <p className="text-sm text-neutral-300 mt-1">
                            Score: {result.score.toFixed(2)}
                          </p>
                          {result.highlights.length > 0 && (
                            <div className="mt-2">
                              {result.highlights.map((highlight, hIndex) => (
                                <span key={hIndex} className="text-xs text-neutral-400 block">
                                  {highlight}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </Card>
              </TabsContent>
            </Tabs>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Folder className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Project Selected
              </h3>
              <p className="text-neutral-400">
                Analyze a project or select an existing context to get started
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
