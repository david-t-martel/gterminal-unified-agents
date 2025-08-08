import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Play,
  Square,
  Settings,
  FileCode,
  CheckCircle,
  AlertCircle,
  Clock,
  Code,
  GitBranch,
  Zap,
  FileSearch,
  RotateCcw
} from 'lucide-react';

interface DiagnosticResult {
  file_path: string;
  line_number: number;
  column?: number;
  severity: string;
  code?: string;
  message: string;
  source: string;
  fixable: boolean;
  suggested_fix?: string;
}

interface AnalysisResponse {
  session_id: string;
  total_issues: number;
  fixable_issues: number;
  issues_by_language: Record<string, number>;
  issues_by_severity: Record<string, number>;
  diagnostics: DiagnosticResult[];
  status: string;
}

interface FixResponse {
  session_id: string;
  success: boolean;
  files_modified: string[];
  diagnostics_fixed: number;
  diagnostics_remaining: number;
  errors: string[];
  status: string;
}

interface AutoClaudeSession {
  session_id: string;
  project_root: string;
  backend_url: string;
  created: string;
}

export function AutoClaudeDashboard() {
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [sessions, setSessions] = useState<AutoClaudeSession[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isFixing, setIsFixing] = useState(false);
  const [projectRoot, setProjectRoot] = useState('/home/david/agents/my-fullstack-agent');
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(['typescript', 'python', 'javascript']);
  const [maxFixes, setMaxFixes] = useState(50);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const response = await fetch('/api/v1/auto-claude/sessions');
      const data = await response.json();
      setSessions(data.sessions || []);
    } catch (err) {
      console.error('Failed to load sessions:', err);
    }
  };

  const analyzeProject = async () => {
    setIsAnalyzing(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('/api/v1/auto-claude/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_root: projectRoot,
          languages: selectedLanguages,
          enable_typescript: selectedLanguages.includes('typescript'),
          enable_ai_fixes: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result: AnalysisResponse = await response.json();
      setAnalysis(result);
      setSuccess(`Analysis completed! Found ${result.total_issues} issues (${result.fixable_issues} fixable)`);
      await loadSessions();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fixIssues = async (preview = false) => {
    if (!analysis) return;

    setIsFixing(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('/api/v1/auto-claude/fix', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: analysis.session_id,
          auto_fix: true,
          preview_only: preview,
          max_fixes: maxFixes,
        }),
      });

      if (!response.ok) {
        throw new Error(`Fix operation failed: ${response.statusText}`);
      }

      const result: FixResponse = await response.json();

      if (preview) {
        setSuccess(`Preview: Would fix ${result.diagnostics_fixed} issues in ${result.files_modified.length} files`);
      } else {
        setSuccess(`Fixed ${result.diagnostics_fixed} issues in ${result.files_modified.length} files!`);
        // Refresh analysis
        await analyzeProject();
      }

      if (result.errors.length > 0) {
        setError(`Some errors occurred: ${result.errors.join(', ')}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Fix operation failed');
    } finally {
      setIsFixing(false);
    }
  };

  const installHooks = async () => {
    try {
      const response = await fetch('/api/v1/auto-claude/install-hooks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_root: projectRoot,
        }),
      });

      if (!response.ok) {
        throw new Error(`Hook installation failed: ${response.statusText}`);
      }

      setSuccess('Pre-commit hooks installed successfully!');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Hook installation failed');
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'error': return 'bg-red-500';
      case 'warning': return 'bg-yellow-500';
      case 'info': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getLanguageIcon = (language: string) => {
    switch (language.toLowerCase()) {
      case 'typescript': return '⚡';
      case 'javascript': return '📜';
      case 'python': return '🐍';
      case 'rust': return '🦀';
      default: return '📄';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Enhanced Auto-Claude</h1>
          <p className="text-neutral-400 mt-1">
            TypeScript-aware code analysis and automated fixing
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button onClick={loadSessions} variant="outline" size="sm">
            <RotateCcw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}

      {/* Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Configuration
          </CardTitle>
          <CardDescription>
            Configure analysis settings and project parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">Project Root</label>
            <input
              type="text"
              value={projectRoot}
              onChange={(e) => setProjectRoot(e.target.value)}
              className="w-full p-2 border border-neutral-700 rounded bg-neutral-800 text-neutral-100"
              placeholder="/path/to/project"
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Languages to Analyze</label>
            <div className="flex flex-wrap gap-2">
              {['typescript', 'javascript', 'python', 'rust'].map((lang) => (
                <Button
                  key={lang}
                  variant={selectedLanguages.includes(lang) ? "default" : "outline"}
                  size="sm"
                  onClick={() => {
                    if (selectedLanguages.includes(lang)) {
                      setSelectedLanguages(selectedLanguages.filter(l => l !== lang));
                    } else {
                      setSelectedLanguages([...selectedLanguages, lang]);
                    }
                  }}
                >
                  {getLanguageIcon(lang)} {lang}
                </Button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">Max Fixes per Session</label>
            <input
              type="number"
              value={maxFixes}
              onChange={(e) => setMaxFixes(parseInt(e.target.value) || 50)}
              className="w-32 p-2 border border-neutral-700 rounded bg-neutral-800 text-neutral-100"
              min="1"
              max="200"
            />
          </div>
        </CardContent>
      </Card>

      {/* Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Zap className="w-5 h-5 mr-2" />
            Actions
          </CardTitle>
          <CardDescription>
            Analyze your project and apply automated fixes
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-3">
            <Button
              onClick={analyzeProject}
              disabled={isAnalyzing}
              className="flex items-center"
            >
              <FileSearch className="w-4 h-4 mr-2" />
              {isAnalyzing ? 'Analyzing...' : 'Analyze Project'}
            </Button>

            {analysis && (
              <>
                <Button
                  onClick={() => fixIssues(true)}
                  disabled={isFixing}
                  variant="outline"
                >
                  <FileCode className="w-4 h-4 mr-2" />
                  Preview Fixes
                </Button>

                <Button
                  onClick={() => fixIssues(false)}
                  disabled={isFixing || analysis.fixable_issues === 0}
                  variant="default"
                >
                  <CheckCircle className="w-4 h-4 mr-2" />
                  {isFixing ? 'Fixing...' : `Apply ${analysis.fixable_issues} Fixes`}
                </Button>
              </>
            )}

            <Button onClick={installHooks} variant="outline">
              <GitBranch className="w-4 h-4 mr-2" />
              Install Pre-commit Hooks
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-6">
          {/* Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <FileCode className="w-5 h-5 mr-2" />
                Analysis Summary
              </CardTitle>
              <CardDescription>Session: {analysis.session_id}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center p-4 bg-neutral-800 rounded-lg">
                  <div className="text-2xl font-bold text-red-400">{analysis.total_issues}</div>
                  <div className="text-sm text-neutral-400">Total Issues</div>
                </div>
                <div className="text-center p-4 bg-neutral-800 rounded-lg">
                  <div className="text-2xl font-bold text-green-400">{analysis.fixable_issues}</div>
                  <div className="text-sm text-neutral-400">Fixable Issues</div>
                </div>
                <div className="text-center p-4 bg-neutral-800 rounded-lg">
                  <div className="text-2xl font-bold text-blue-400">
                    {Math.round((analysis.fixable_issues / analysis.total_issues) * 100)}%
                  </div>
                  <div className="text-sm text-neutral-400">Auto-fixable</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Language Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Issues by Language</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(analysis.issues_by_language).map(([language, count]) => (
                  <div key={language} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <span className="mr-2">{getLanguageIcon(language)}</span>
                      <span className="capitalize">{language}</span>
                    </div>
                    <Badge variant="secondary">{count}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Severity Breakdown */}
          <Card>
            <CardHeader>
              <CardTitle>Issues by Severity</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {Object.entries(analysis.issues_by_severity).map(([severity, count]) => (
                  <div key={severity} className="flex items-center justify-between">
                    <div className="flex items-center">
                      <div className={`w-3 h-3 rounded-full mr-2 ${getSeverityColor(severity)}`}></div>
                      <span className="capitalize">{severity}</span>
                    </div>
                    <Badge variant="secondary">{count}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Sample Issues */}
          <Card>
            <CardHeader>
              <CardTitle>Sample Issues</CardTitle>
              <CardDescription>
                Showing first 10 issues found (total: {analysis.diagnostics.length})
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {analysis.diagnostics.slice(0, 10).map((diagnostic, index) => (
                  <div key={index} className="border border-neutral-700 rounded-lg p-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <Badge variant={diagnostic.severity === 'error' ? 'destructive' : 'secondary'}>
                            {diagnostic.severity}
                          </Badge>
                          <Badge variant="outline">{diagnostic.source}</Badge>
                          {diagnostic.fixable && (
                            <Badge variant="default" className="bg-green-600">
                              <CheckCircle className="w-3 h-3 mr-1" />
                              Fixable
                            </Badge>
                          )}
                        </div>
                        <div className="text-sm text-neutral-300 mb-1">
                          {diagnostic.file_path.split('/').pop()}:{diagnostic.line_number}
                          {diagnostic.column && `:${diagnostic.column}`}
                        </div>
                        <div className="text-sm">{diagnostic.message}</div>
                        {diagnostic.code && (
                          <div className="text-xs text-neutral-500 mt-1">
                            Code: {diagnostic.code}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Active Sessions */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Clock className="w-5 h-5 mr-2" />
            Active Sessions
          </CardTitle>
          <CardDescription>
            {sessions.length} active auto-claude sessions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sessions.length === 0 ? (
            <div className="text-center py-8 text-neutral-400">
              No active sessions
            </div>
          ) : (
            <div className="space-y-3">
              {sessions.map((session) => (
                <div key={session.session_id} className="flex items-center justify-between p-3 border border-neutral-700 rounded-lg">
                  <div>
                    <div className="font-medium">{session.session_id}</div>
                    <div className="text-sm text-neutral-400">{session.project_root}</div>
                  </div>
                  <Badge variant="outline">{session.created}</Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
