import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Play,
  Square,
  Download,
  Copy,
  File,
  Folder,
  RefreshCw,
  Sparkles,
  AlertCircle,
  CheckCircle,
  Clock,
  Eye,
  Code,
  Lightbulb,
  Shield,
  Zap,
  Settings
} from 'lucide-react';
import {
  LanguageType,
  CodeGenerationRequest,
  CodeGenerationResponse,
  GeneratedFile,
  ProAgentInsight,
  ProjectContext,
  GenerationProgressEvent
} from '@/types/api';
import { apiService } from '@/services/api';
import { getWebSocketService } from '@/services/websocket';

interface GeneratedFileViewerProps {
  file: GeneratedFile;
  onCopy: (content: string) => void;
}

const GeneratedFileViewer: React.FC<GeneratedFileViewerProps> = ({ file, onCopy }) => {
  const getLanguageIcon = (language: LanguageType) => {
    // Simple icon mapping - in a real app you might use syntax highlighting
    return <Code className="w-4 h-4" />;
  };

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

  return (
    <Card className="border-neutral-700 bg-neutral-800/50">
      <div className="flex items-center justify-between p-4 border-b border-neutral-700">
        <div className="flex items-center gap-2">
          {getLanguageIcon(file.language)}
          <span className="font-medium text-neutral-100">{file.path}</span>
          <Badge className={`${getLanguageColor(file.language)} px-2 py-0`}>
            {file.language}
          </Badge>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onCopy(file.content)}
            className="text-neutral-400 hover:text-neutral-200"
          >
            <Copy className="w-4 h-4" />
          </Button>
        </div>
      </div>

      <div className="p-4">
        <div className="mb-3">
          <h4 className="text-sm font-medium text-neutral-300 mb-1">Purpose</h4>
          <p className="text-sm text-neutral-400">{file.purpose}</p>
        </div>

        {file.dependencies && file.dependencies.length > 0 && (
          <div className="mb-3">
            <h4 className="text-sm font-medium text-neutral-300 mb-2">Dependencies</h4>
            <div className="flex flex-wrap gap-1">
              {file.dependencies.map((dep, index) => (
                <Badge key={index} variant="outline" className="text-xs">
                  {dep}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <div>
          <h4 className="text-sm font-medium text-neutral-300 mb-2">Code</h4>
          <ScrollArea className="h-64">
            <pre className="text-xs text-neutral-300 bg-neutral-900 p-3 rounded-md overflow-x-auto">
              <code>{file.content}</code>
            </pre>
          </ScrollArea>
        </div>
      </div>
    </Card>
  );
};

interface InsightCardProps {
  insight: ProAgentInsight;
}

const InsightCard: React.FC<InsightCardProps> = ({ insight }) => {
  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'best_practice': return <Lightbulb className="w-4 h-4" />;
      case 'optimization': return <Zap className="w-4 h-4" />;
      case 'security': return <Shield className="w-4 h-4" />;
      case 'architecture': return <Code className="w-4 h-4" />;
      default: return <Lightbulb className="w-4 h-4" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-red-500/20 text-red-300 border-red-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 'low': return 'bg-green-500/20 text-green-300 border-green-500/30';
      default: return 'bg-neutral-500/20 text-neutral-300 border-neutral-500/30';
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'best_practice': return 'text-blue-400';
      case 'optimization': return 'text-green-400';
      case 'security': return 'text-red-400';
      case 'architecture': return 'text-purple-400';
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
        <Badge className={`${getImpactColor(insight.impact)} border`}>
          {insight.impact} impact
        </Badge>
      </div>

      <p className="text-neutral-300 text-sm mb-3">{insight.description}</p>

      {insight.implementation && (
        <div className="bg-neutral-700/50 p-3 rounded-md">
          <h4 className="text-xs font-semibold text-neutral-200 mb-1">Implementation:</h4>
          <p className="text-xs text-neutral-300">{insight.implementation}</p>
        </div>
      )}
    </Card>
  );
};

export const CodeGeneration: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [language, setLanguage] = useState<LanguageType>(LanguageType.PYTHON);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentGeneration, setCurrentGeneration] = useState<CodeGenerationResponse | null>(null);
  const [progress, setProgress] = useState<GenerationProgressEvent | null>(null);
  const [activeTab, setActiveTab] = useState('generate');
  const [generations, setGenerations] = useState<CodeGenerationResponse[]>([]);
  const [selectedContext, setSelectedContext] = useState<ProjectContext | null>(null);
  const [contexts, setContexts] = useState<ProjectContext[]>([]);

  // Generation settings
  const [maxFiles, setMaxFiles] = useState(10);
  const [maxLinesPerFile, setMaxLinesPerFile] = useState(500);
  const [includeTests, setIncludeTests] = useState(true);
  const [includeDocumentation, setIncludeDocumentation] = useState(true);
  const [codeStyle, setCodeStyle] = useState('clean');
  const [framework, setFramework] = useState('');

  const wsRef = useRef<any>(null);

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

  const loadGenerations = useCallback(async () => {
    try {
      const response = await apiService.listGenerations();
      if (response.success) {
        setGenerations(response.data);
      }
    } catch (err) {
      console.error('Failed to load generations:', err);
    }
  }, []);

  const generateCode = useCallback(async () => {
    if (!prompt.trim()) return;

    try {
      setGenerating(true);
      setError(null);
      setProgress(null);

      const request: CodeGenerationRequest = {
        prompt: prompt.trim(),
        language,
        context: selectedContext || undefined,
        constraints: {
          maxFiles,
          maxLinesPerFile,
          includeTests,
          includeDocumentation
        },
        preferences: {
          codeStyle,
          framework: framework || undefined,
          patterns: ['solid', 'dry', 'clean-code']
        }
      };

      const response = await apiService.generateCode(request);
      if (response.success) {
        setCurrentGeneration(response.data);

        // Subscribe to WebSocket updates for this generation
        if (wsRef.current) {
          wsRef.current.subscribeToGeneration(response.data.requestId);
        }

        await loadGenerations();
      } else {
        setError(response.error?.message || 'Generation failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed');
    } finally {
      setGenerating(false);
    }
  }, [prompt, language, selectedContext, maxFiles, maxLinesPerFile, includeTests, includeDocumentation, codeStyle, framework, loadGenerations]);

  const cancelGeneration = useCallback(async () => {
    if (!currentGeneration) return;

    try {
      const response = await apiService.cancelGeneration(currentGeneration.requestId);
      if (response.success) {
        setGenerating(false);
        setProgress(null);

        if (wsRef.current) {
          wsRef.current.unsubscribeFromGeneration(currentGeneration.requestId);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel generation');
    }
  }, [currentGeneration]);

  const copyToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      // Could show a toast notification here
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  }, []);

  const downloadGeneration = useCallback(() => {
    if (!currentGeneration) return;

    const zip = currentGeneration.generatedFiles.map(file => ({
      name: file.path.replace(/^\//, ''),
      content: file.content
    }));

    // In a real app, you'd use a library like JSZip to create a proper zip file
    const blob = new Blob([JSON.stringify(zip, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generation-${currentGeneration.requestId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [currentGeneration]);

  // Set up WebSocket for real-time updates
  useEffect(() => {
    const wsBaseUrl = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsBaseUrl}//${window.location.host}/api/v1/generation/stream`;

    try {
      const ws = getWebSocketService(wsUrl);
      wsRef.current = ws;

      const handleGenerationProgress = (data: GenerationProgressEvent) => {
        setProgress(data);

        // Update current generation if it matches
        if (currentGeneration?.requestId === data.requestId) {
          // Update insights as they come in
          if (data.insights) {
            setCurrentGeneration(prev => prev ? {
              ...prev,
              proAgentInsights: [...prev.proAgentInsights, ...data.insights!]
            } : null);
          }
        }
      };

      ws.on('generation_progress', handleGenerationProgress);
      ws.connect().catch(console.error);

      return () => {
        ws.off('generation_progress', handleGenerationProgress);
        ws.disconnect();
      };
    } catch (err) {
      console.warn('WebSocket not available for real-time updates');
    }
  }, [currentGeneration]);

  useEffect(() => {
    loadContexts();
    loadGenerations();
  }, [loadContexts, loadGenerations]);

  const formatExecutionTime = (time: number) => {
    if (time < 1000) return `${time}ms`;
    return `${(time / 1000).toFixed(1)}s`;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Code Generation</h1>
          <p className="text-neutral-400">Generate high-quality code with AI assistance</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadGenerations} variant="outline" size="sm">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
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
          <TabsTrigger value="generate">Generate</TabsTrigger>
          <TabsTrigger value="results">Results</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>

        <TabsContent value="generate" className="mt-6 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Generation Form */}
            <div className="lg:col-span-2 space-y-4">
              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <h3 className="font-semibold text-neutral-100 mb-4">Generation Request</h3>

                <div className="space-y-4">
                  <div>
                    <label className="text-sm font-medium text-neutral-300 block mb-2">
                      Prompt
                    </label>
                    <Textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      placeholder="Describe what code you want to generate..."
                      rows={4}
                      className="bg-neutral-700 border-neutral-600"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm font-medium text-neutral-300 block mb-2">
                        Language
                      </label>
                      <Select value={language} onValueChange={(value) => setLanguage(value as LanguageType)}>
                        <SelectTrigger className="bg-neutral-700 border-neutral-600">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.values(LanguageType).map((lang) => (
                            <SelectItem key={lang} value={lang}>
                              {lang.charAt(0).toUpperCase() + lang.slice(1)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <label className="text-sm font-medium text-neutral-300 block mb-2">
                        Context
                      </label>
                      <Select
                        value={selectedContext?.id || "none"}
                        onValueChange={(value) => {
                          if (value === "none") {
                            setSelectedContext(null);
                          } else {
                            const context = contexts.find(c => c.id === value);
                            setSelectedContext(context || null);
                          }
                        }}
                      >
                        <SelectTrigger className="bg-neutral-700 border-neutral-600">
                          <SelectValue placeholder="Select context" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="none">No context</SelectItem>
                          {contexts.map((context) => (
                            <SelectItem key={context.id} value={context.id}>
                              {context.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Advanced Settings */}
                  <details className="group">
                    <summary className="flex items-center gap-2 cursor-pointer text-neutral-300 hover:text-neutral-100">
                      <Settings className="w-4 h-4" />
                      <span className="text-sm font-medium">Advanced Settings</span>
                    </summary>

                    <div className="mt-4 space-y-4 pl-6">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium text-neutral-300 block mb-2">
                            Max Files
                          </label>
                          <Input
                            type="number"
                            value={maxFiles}
                            onChange={(e) => setMaxFiles(parseInt(e.target.value) || 10)}
                            min={1}
                            max={50}
                            className="bg-neutral-700 border-neutral-600"
                          />
                        </div>

                        <div>
                          <label className="text-sm font-medium text-neutral-300 block mb-2">
                            Max Lines per File
                          </label>
                          <Input
                            type="number"
                            value={maxLinesPerFile}
                            onChange={(e) => setMaxLinesPerFile(parseInt(e.target.value) || 500)}
                            min={10}
                            max={2000}
                            className="bg-neutral-700 border-neutral-600"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium text-neutral-300 block mb-2">
                            Code Style
                          </label>
                          <Select value={codeStyle} onValueChange={setCodeStyle}>
                            <SelectTrigger className="bg-neutral-700 border-neutral-600">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="clean">Clean Code</SelectItem>
                              <SelectItem value="minimal">Minimal</SelectItem>
                              <SelectItem value="verbose">Verbose</SelectItem>
                              <SelectItem value="enterprise">Enterprise</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div>
                          <label className="text-sm font-medium text-neutral-300 block mb-2">
                            Framework (Optional)
                          </label>
                          <Input
                            value={framework}
                            onChange={(e) => setFramework(e.target.value)}
                            placeholder="e.g., React, FastAPI, Django"
                            className="bg-neutral-700 border-neutral-600"
                          />
                        </div>
                      </div>

                      <div className="flex gap-4">
                        <label className="flex items-center gap-2 text-sm text-neutral-300">
                          <input
                            type="checkbox"
                            checked={includeTests}
                            onChange={(e) => setIncludeTests(e.target.checked)}
                            className="rounded"
                          />
                          Include Tests
                        </label>

                        <label className="flex items-center gap-2 text-sm text-neutral-300">
                          <input
                            type="checkbox"
                            checked={includeDocumentation}
                            onChange={(e) => setIncludeDocumentation(e.target.checked)}
                            className="rounded"
                          />
                          Include Documentation
                        </label>
                      </div>
                    </div>
                  </details>

                  <div className="flex gap-2 pt-4">
                    <Button
                      onClick={generateCode}
                      disabled={generating || !prompt.trim()}
                      className="flex-1 bg-blue-600 hover:bg-blue-700"
                    >
                      {generating ? (
                        <>
                          <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                          Generating...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4 mr-2" />
                          Generate Code
                        </>
                      )}
                    </Button>

                    {generating && (
                      <Button
                        onClick={cancelGeneration}
                        variant="outline"
                        className="border-red-500/30 text-red-300 hover:bg-red-500/20"
                      >
                        <Square className="w-4 h-4 mr-2" />
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>
              </Card>

              {/* Progress Indicator */}
              {progress && (
                <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-neutral-100">Generation Progress</h3>
                    <span className="text-sm text-neutral-400">
                      {Math.round(progress.progress)}%
                    </span>
                  </div>

                  <div className="space-y-3">
                    <div className="w-full bg-neutral-700 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress.progress}%` }}
                      />
                    </div>

                    <div className="flex justify-between text-sm text-neutral-400">
                      <span>
                        {progress.completedFiles} / {progress.totalFiles} files
                      </span>
                      {progress.currentFile && (
                        <span>Current: {progress.currentFile}</span>
                      )}
                    </div>
                  </div>
                </Card>
              )}
            </div>

            {/* Context Info */}
            <div>
              {selectedContext && (
                <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                  <h3 className="font-semibold text-neutral-100 mb-3">Selected Context</h3>

                  <div className="space-y-3">
                    <div>
                      <h4 className="text-sm font-medium text-neutral-300">{selectedContext.name}</h4>
                      <p className="text-xs text-neutral-400 mt-1">
                        {selectedContext.structure.totalFiles} files, {selectedContext.dependencies.length} dependencies
                      </p>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      <Badge className="text-xs">{selectedContext.language}</Badge>
                      {selectedContext.dependencies.slice(0, 3).map((dep, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {dep.name}
                        </Badge>
                      ))}
                      {selectedContext.dependencies.length > 3 && (
                        <Badge variant="outline" className="text-xs">
                          +{selectedContext.dependencies.length - 3} more
                        </Badge>
                      )}
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="results" className="mt-6">
          {currentGeneration ? (
            <div className="space-y-6">
              <Card className="p-4 border-neutral-700 bg-neutral-800/50">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="font-semibold text-neutral-100">Generation Results</h3>
                    <p className="text-sm text-neutral-400">
                      {currentGeneration.generatedFiles.length} files, {currentGeneration.metadata.totalLines} lines
                      • Generated in {formatExecutionTime(currentGeneration.metadata.generationTime)}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button onClick={downloadGeneration} variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>

                {currentGeneration.executionPlan && (
                  <div className="mb-4 p-3 bg-neutral-700/50 rounded-md">
                    <h4 className="text-sm font-semibold text-neutral-200 mb-2">Execution Plan</h4>
                    <div className="space-y-1">
                      {currentGeneration.executionPlan.steps.map((step, index) => (
                        <div key={index} className="flex items-center gap-2 text-xs text-neutral-300">
                          <CheckCircle className="w-3 h-3 text-green-400" />
                          {step}
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-neutral-400 mt-2">
                      Estimated time: {currentGeneration.executionPlan.estimatedTime}
                    </p>
                  </div>
                )}
              </Card>

              <div className="grid gap-4">
                {currentGeneration.generatedFiles.map((file, index) => (
                  <GeneratedFileViewer
                    key={index}
                    file={file}
                    onCopy={copyToClipboard}
                  />
                ))}
              </div>
            </div>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Sparkles className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Generation Yet
              </h3>
              <p className="text-neutral-400">
                Generate some code to see the results here
              </p>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="insights" className="mt-6">
          {currentGeneration && currentGeneration.proAgentInsights.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2">
              {currentGeneration.proAgentInsights.map((insight, index) => (
                <InsightCard key={index} insight={insight} />
              ))}
            </div>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Lightbulb className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Insights Available
              </h3>
              <p className="text-neutral-400">
                Generate code to receive AI insights and recommendations
              </p>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="history" className="mt-6">
          {generations.length > 0 ? (
            <div className="space-y-4">
              {generations.map((generation) => (
                <Card
                  key={generation.requestId}
                  className={`p-4 border-neutral-700 bg-neutral-800/50 cursor-pointer transition-colors ${
                    currentGeneration?.requestId === generation.requestId
                      ? 'ring-2 ring-blue-500/50'
                      : 'hover:bg-neutral-800'
                  }`}
                  onClick={() => setCurrentGeneration(generation)}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-medium text-neutral-100">
                        Generation {generation.requestId.slice(-8)}
                      </h3>
                      <p className="text-sm text-neutral-400 mt-1">
                        {generation.generatedFiles.length} files • {generation.metadata.totalLines} lines
                        • {formatExecutionTime(generation.metadata.generationTime)}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          setCurrentGeneration(generation);
                          setActiveTab('results');
                        }}
                      >
                        <Eye className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Clock className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Generation History
              </h3>
              <p className="text-neutral-400">
                Your generation history will appear here
              </p>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};
