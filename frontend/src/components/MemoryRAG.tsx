import React, { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Search,
  Plus,
  Database,
  Upload,
  Download,
  Trash2,
  RefreshCw,
  Filter,
  SortAsc,
  SortDesc,
  Tag,
  Calendar,
  File,
  Code,
  FileText,
  MessageSquare,
  Brain,
  Sparkles,
  Target,
  Clock
} from 'lucide-react';
import {
  MemoryEmbedRequest,
  MemorySearchRequest,
  MemorySearchResponse,
  MemorySearchResult,
  LanguageType
} from '@/types/api';
import { apiService } from '@/services/api';

interface SearchResultCardProps {
  result: MemorySearchResult;
  onDelete: (embeddingId: string) => void;
}

const SearchResultCard: React.FC<SearchResultCardProps> = ({ result, onDelete }) => {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'code': return <Code className="w-4 h-4" />;
      case 'documentation': return <FileText className="w-4 h-4" />;
      case 'conversation': return <MessageSquare className="w-4 h-4" />;
      case 'file': return <File className="w-4 h-4" />;
      default: return <Database className="w-4 h-4" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'code': return 'text-blue-400';
      case 'documentation': return 'text-green-400';
      case 'conversation': return 'text-purple-400';
      case 'file': return 'text-orange-400';
      default: return 'text-neutral-400';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400';
    if (score >= 0.6) return 'text-yellow-400';
    if (score >= 0.4) return 'text-orange-400';
    return 'text-red-400';
  };

  const formatScore = (score: number) => {
    return `${(score * 100).toFixed(1)}%`;
  };

  const getLanguageColor = (language?: string) => {
    if (!language) return 'bg-neutral-500/20 text-neutral-300';

    switch (language.toLowerCase()) {
      case 'python': return 'bg-blue-500/20 text-blue-300';
      case 'typescript': return 'bg-blue-600/20 text-blue-400';
      case 'javascript': return 'bg-yellow-500/20 text-yellow-300';
      case 'rust': return 'bg-orange-500/20 text-orange-300';
      case 'go': return 'bg-cyan-500/20 text-cyan-300';
      case 'java': return 'bg-red-500/20 text-red-300';
      default: return 'bg-neutral-500/20 text-neutral-300';
    }
  };

  return (
    <Card className="p-4 border-neutral-700 bg-neutral-800/50 hover:bg-neutral-800 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={getTypeColor(result.metadata.type)}>
            {getTypeIcon(result.metadata.type)}
          </div>
          <h3 className="font-medium text-neutral-100">
            {result.metadata.source || 'Unknown Source'}
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <Badge className={`${getScoreColor(result.score)} bg-transparent border-current`}>
            {formatScore(result.score)}
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => onDelete(result.embeddingId)}
            className="h-6 w-6 p-0 text-neutral-400 hover:text-red-400"
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <ScrollArea className="h-20">
            <p className="text-sm text-neutral-300">
              {result.content.length > 200
                ? `${result.content.substring(0, 200)}...`
                : result.content
              }
            </p>
          </ScrollArea>
        </div>

        {result.highlights && result.highlights.length > 0 && (
          <div>
            <h4 className="text-xs font-semibold text-neutral-400 mb-1">Highlights:</h4>
            <div className="space-y-1">
              {result.highlights.slice(0, 2).map((highlight, index) => (
                <p key={index} className="text-xs text-yellow-300 bg-yellow-500/10 p-1 rounded">
                  {highlight}
                </p>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              {result.metadata.type}
            </Badge>
            {result.metadata.language && (
              <Badge className={`${getLanguageColor(result.metadata.language)} text-xs`}>
                {result.metadata.language}
              </Badge>
            )}
          </div>

          {result.metadata.context?.createdAt && (
            <span className="text-xs text-neutral-500">
              {new Date(result.metadata.context.createdAt).toLocaleDateString()}
            </span>
          )}
        </div>
      </div>
    </Card>
  );
};

interface AddMemoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (request: MemoryEmbedRequest) => void;
}

const AddMemoryModal: React.FC<AddMemoryModalProps> = ({ isOpen, onClose, onAdd }) => {
  const [content, setContent] = useState('');
  const [type, setType] = useState<'code' | 'documentation' | 'conversation' | 'file'>('code');
  const [source, setSource] = useState('');
  const [language, setLanguage] = useState<LanguageType | ''>('');
  const [tags, setTags] = useState('');

  const handleSubmit = () => {
    if (!content.trim() || !source.trim()) return;

    const request: MemoryEmbedRequest = {
      content: content.trim(),
      metadata: {
        type,
        source: source.trim(),
        language: language || undefined,
        context: {
          createdAt: new Date().toISOString(),
          addedBy: 'user'
        }
      },
      tags: tags.split(',').map(tag => tag.trim()).filter(Boolean)
    };

    onAdd(request);

    // Reset form
    setContent('');
    setSource('');
    setLanguage('');
    setTags('');
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <Card className="w-full max-w-2xl mx-4 p-6 border-neutral-700 bg-neutral-800">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-neutral-100">Add to Memory</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>
            ×
          </Button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm font-medium text-neutral-300 block mb-2">
              Content
            </label>
            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Enter the content to embed..."
              rows={6}
              className="bg-neutral-700 border-neutral-600"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-neutral-300 block mb-2">
                Type
              </label>
              <Select value={type} onValueChange={(value) => setType(value as any)}>
                <SelectTrigger className="bg-neutral-700 border-neutral-600">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="code">Code</SelectItem>
                  <SelectItem value="documentation">Documentation</SelectItem>
                  <SelectItem value="conversation">Conversation</SelectItem>
                  <SelectItem value="file">File</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-neutral-300 block mb-2">
                Source
              </label>
              <Input
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder="e.g., main.py, README.md"
                className="bg-neutral-700 border-neutral-600"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-neutral-300 block mb-2">
                Language (Optional)
              </label>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="bg-neutral-700 border-neutral-600">
                  <SelectValue placeholder="Select language" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">None</SelectItem>
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
                Tags (Optional)
              </label>
              <Input
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="tag1, tag2, tag3"
                className="bg-neutral-700 border-neutral-600"
              />
            </div>
          </div>

          <div className="flex gap-2 pt-4">
            <Button onClick={handleSubmit} disabled={!content.trim() || !source.trim()}>
              <Plus className="w-4 h-4 mr-2" />
              Add to Memory
            </Button>
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export const MemoryRAG: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<MemorySearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('search');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  // Search filters
  const [topK, setTopK] = useState(20);
  const [threshold, setThreshold] = useState(0.1);
  const [typeFilter, setTypeFilter] = useState<string[]>([]);
  const [languageFilter, setLanguageFilter] = useState<LanguageType[]>([]);
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<'score' | 'date'>('score');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Recent searches
  const [recentSearches, setRecentSearches] = useState<string[]>([]);

  // Stats
  const [searchStats, setSearchStats] = useState({
    total: 0,
    took: 0,
    avgScore: 0
  });

  const searchMemory = useCallback(async () => {
    if (!searchQuery.trim()) return;

    try {
      setLoading(true);
      setError(null);

      const request: MemorySearchRequest = {
        query: searchQuery.trim(),
        topK,
        threshold,
        filters: {
          type: typeFilter.length > 0 ? typeFilter : undefined,
          language: languageFilter.length > 0 ? languageFilter : undefined,
          tags: tagFilter.length > 0 ? tagFilter : undefined
        }
      };

      const response = await apiService.searchMemory(request);
      if (response.success) {
        let results = response.data.results;

        // Apply sorting
        results.sort((a, b) => {
          if (sortBy === 'score') {
            return sortOrder === 'desc' ? b.score - a.score : a.score - b.score;
          } else {
            const dateA = new Date(a.metadata.context?.createdAt || 0).getTime();
            const dateB = new Date(b.metadata.context?.createdAt || 0).getTime();
            return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
          }
        });

        setSearchResults(results);
        setSearchStats({
          total: response.data.total,
          took: response.data.took,
          avgScore: results.length > 0 ? results.reduce((sum, r) => sum + r.score, 0) / results.length : 0
        });

        // Add to recent searches
        setRecentSearches(prev => {
          const updated = [searchQuery, ...prev.filter(q => q !== searchQuery)];
          return updated.slice(0, 10);
        });
      } else {
        setError(response.error?.message || 'Search failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }, [searchQuery, topK, threshold, typeFilter, languageFilter, tagFilter, sortBy, sortOrder]);

  const addMemory = useCallback(async (request: MemoryEmbedRequest) => {
    try {
      setLoading(true);
      const response = await apiService.embedMemory(request);
      if (response.success) {
        // Optionally refresh search results if there's an active search
        if (searchQuery.trim()) {
          await searchMemory();
        }
      } else {
        setError(response.error?.message || 'Failed to add memory');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add memory');
    } finally {
      setLoading(false);
    }
  }, [searchQuery, searchMemory]);

  const deleteMemory = useCallback(async (embeddingId: string) => {
    try {
      const response = await apiService.deleteMemory(embeddingId);
      if (response.success) {
        setSearchResults(prev => prev.filter(r => r.embeddingId !== embeddingId));
        setSearchStats(prev => ({ ...prev, total: prev.total - 1 }));
      } else {
        setError(response.error?.message || 'Failed to delete memory');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete memory');
    }
  }, []);

  const clearFilters = useCallback(() => {
    setTypeFilter([]);
    setLanguageFilter([]);
    setTagFilter([]);
    setThreshold(0.1);
    setTopK(20);
  }, []);

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-100">Memory RAG</h1>
          <p className="text-neutral-400">Search and manage semantic memory with AI embeddings</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setIsAddModalOpen(true)} variant="outline" size="sm">
            <Plus className="w-4 h-4 mr-2" />
            Add Memory
          </Button>
        </div>
      </div>

      {error && (
        <Card className="p-4 border-red-500/30 bg-red-500/10">
          <div className="flex items-center gap-2 text-red-300">
            <Target className="w-4 h-4" />
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
        <TabsList className="grid w-full grid-cols-3 bg-neutral-800">
          <TabsTrigger value="search">Search</TabsTrigger>
          <TabsTrigger value="filters">Filters</TabsTrigger>
          <TabsTrigger value="stats">Statistics</TabsTrigger>
        </TabsList>

        <TabsContent value="search" className="mt-6 space-y-6">
          {/* Search Interface */}
          <Card className="p-4 border-neutral-700 bg-neutral-800/50">
            <div className="space-y-4">
              <div className="flex gap-4">
                <div className="flex-1">
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search memory with semantic similarity..."
                    className="bg-neutral-700 border-neutral-600"
                    onKeyPress={(e) => e.key === 'Enter' && searchMemory()}
                  />
                </div>
                <Button
                  onClick={searchMemory}
                  disabled={loading || !searchQuery.trim()}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Search className="w-4 h-4 mr-2" />
                  )}
                  Search
                </Button>
              </div>

              {/* Quick Actions */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <label className="text-sm text-neutral-400">Results:</label>
                  <Select value={topK.toString()} onValueChange={(value) => setTopK(parseInt(value))}>
                    <SelectTrigger className="w-20 bg-neutral-700 border-neutral-600">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="10">10</SelectItem>
                      <SelectItem value="20">20</SelectItem>
                      <SelectItem value="50">50</SelectItem>
                      <SelectItem value="100">100</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center gap-2">
                  <label className="text-sm text-neutral-400">Sort:</label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (sortBy === 'score') {
                        setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
                      } else {
                        setSortBy('score');
                        setSortOrder('desc');
                      }
                    }}
                    className={sortBy === 'score' ? 'bg-blue-500/20' : ''}
                  >
                    <Brain className="w-3 h-3 mr-1" />
                    Score
                    {sortBy === 'score' && (sortOrder === 'desc' ? <SortDesc className="w-3 h-3 ml-1" /> : <SortAsc className="w-3 h-3 ml-1" />)}
                  </Button>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (sortBy === 'date') {
                        setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
                      } else {
                        setSortBy('date');
                        setSortOrder('desc');
                      }
                    }}
                    className={sortBy === 'date' ? 'bg-blue-500/20' : ''}
                  >
                    <Calendar className="w-3 h-3 mr-1" />
                    Date
                    {sortBy === 'date' && (sortOrder === 'desc' ? <SortDesc className="w-3 h-3 ml-1" /> : <SortAsc className="w-3 h-3 ml-1" />)}
                  </Button>
                </div>

                {(typeFilter.length > 0 || languageFilter.length > 0 || tagFilter.length > 0) && (
                  <Button variant="outline" size="sm" onClick={clearFilters}>
                    <Filter className="w-3 h-3 mr-1" />
                    Clear Filters
                  </Button>
                )}
              </div>

              {/* Recent Searches */}
              {recentSearches.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-neutral-300 mb-2">Recent Searches</h4>
                  <div className="flex flex-wrap gap-1">
                    {recentSearches.slice(0, 5).map((query, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        onClick={() => setSearchQuery(query)}
                        className="text-xs h-6"
                      >
                        {query}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-neutral-100">Search Results</h3>
                <div className="flex items-center gap-4 text-sm text-neutral-400">
                  <span>{searchStats.total} results</span>
                  <span>{formatTime(searchStats.took)}</span>
                  <span>Avg: {(searchStats.avgScore * 100).toFixed(1)}%</span>
                </div>
              </div>

              <ScrollArea className="h-96">
                <div className="grid gap-4">
                  {searchResults.map((result) => (
                    <SearchResultCard
                      key={result.embeddingId}
                      result={result}
                      onDelete={deleteMemory}
                    />
                  ))}
                </div>
              </ScrollArea>
            </Card>
          )}

          {searchQuery && searchResults.length === 0 && !loading && (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Search className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                No Results Found
              </h3>
              <p className="text-neutral-400">
                Try adjusting your search query or filters
              </p>
            </Card>
          )}

          {!searchQuery && (
            <Card className="p-8 text-center border-neutral-700 bg-neutral-800/50">
              <Brain className="w-12 h-12 mx-auto text-neutral-400 mb-4" />
              <h3 className="text-lg font-semibold text-neutral-300 mb-2">
                Semantic Memory Search
              </h3>
              <p className="text-neutral-400">
                Enter a query to search through embedded knowledge using AI similarity
              </p>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="filters" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <h3 className="font-semibold text-neutral-100 mb-3">Content Type</h3>
              <div className="space-y-2">
                {['code', 'documentation', 'conversation', 'file'].map((type) => (
                  <label key={type} className="flex items-center gap-2 text-sm text-neutral-300">
                    <input
                      type="checkbox"
                      checked={typeFilter.includes(type)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setTypeFilter(prev => [...prev, type]);
                        } else {
                          setTypeFilter(prev => prev.filter(t => t !== type));
                        }
                      }}
                      className="rounded"
                    />
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </label>
                ))}
              </div>
            </Card>

            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <h3 className="font-semibold text-neutral-100 mb-3">Programming Language</h3>
              <div className="space-y-2">
                {Object.values(LanguageType).map((lang) => (
                  <label key={lang} className="flex items-center gap-2 text-sm text-neutral-300">
                    <input
                      type="checkbox"
                      checked={languageFilter.includes(lang)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setLanguageFilter(prev => [...prev, lang]);
                        } else {
                          setLanguageFilter(prev => prev.filter(l => l !== lang));
                        }
                      }}
                      className="rounded"
                    />
                    {lang.charAt(0).toUpperCase() + lang.slice(1)}
                  </label>
                ))}
              </div>
            </Card>

            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <h3 className="font-semibold text-neutral-100 mb-3">Search Parameters</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-neutral-300 block mb-2">
                    Similarity Threshold: {(threshold * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-neutral-300 block mb-2">
                    Max Results
                  </label>
                  <Input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value) || 20)}
                    min={1}
                    max={200}
                    className="bg-neutral-700 border-neutral-600"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium text-neutral-300 block mb-2">
                    Tags (comma-separated)
                  </label>
                  <Input
                    value={tagFilter.join(', ')}
                    onChange={(e) => setTagFilter(e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
                    placeholder="tag1, tag2, tag3"
                    className="bg-neutral-700 border-neutral-600"
                  />
                </div>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="stats" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <div className="flex items-center gap-2 mb-2">
                <Database className="w-5 h-5 text-blue-400" />
                <h3 className="font-semibold text-neutral-100">Total Results</h3>
              </div>
              <p className="text-2xl font-bold text-neutral-100">{searchStats.total}</p>
              <p className="text-sm text-neutral-400">from last search</p>
            </Card>

            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-5 h-5 text-green-400" />
                <h3 className="font-semibold text-neutral-100">Search Time</h3>
              </div>
              <p className="text-2xl font-bold text-neutral-100">{formatTime(searchStats.took)}</p>
              <p className="text-sm text-neutral-400">response time</p>
            </Card>

            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-5 h-5 text-purple-400" />
                <h3 className="font-semibold text-neutral-100">Avg Score</h3>
              </div>
              <p className="text-2xl font-bold text-neutral-100">
                {(searchStats.avgScore * 100).toFixed(1)}%
              </p>
              <p className="text-sm text-neutral-400">similarity score</p>
            </Card>

            <Card className="p-4 border-neutral-700 bg-neutral-800/50">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-5 h-5 text-yellow-400" />
                <h3 className="font-semibold text-neutral-100">Recent Searches</h3>
              </div>
              <p className="text-2xl font-bold text-neutral-100">{recentSearches.length}</p>
              <p className="text-sm text-neutral-400">in session</p>
            </Card>
          </div>

          {/* Recent Search History */}
          {recentSearches.length > 0 && (
            <Card className="p-4 border-neutral-700 bg-neutral-800/50 mt-6">
              <h3 className="font-semibold text-neutral-100 mb-3">Search History</h3>
              <div className="space-y-2">
                {recentSearches.map((query, index) => (
                  <div key={index} className="flex items-center justify-between p-2 bg-neutral-700/50 rounded">
                    <span className="text-neutral-300">{query}</span>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setSearchQuery(query);
                        setActiveTab('search');
                      }}
                      className="text-neutral-400 hover:text-neutral-200"
                    >
                      Search Again
                    </Button>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <AddMemoryModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onAdd={addMemory}
      />
    </div>
  );
};
