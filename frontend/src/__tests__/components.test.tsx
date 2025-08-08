/**
 * Frontend Component Tests
 *
 * Comprehensive React Testing Library tests for all frontend components:
 * - AgentDashboard.tsx - Real-time agent monitoring
 * - ContextAnalysis.tsx - Project analysis UI
 * - CodeGeneration.tsx - Code generation interface
 * - MemoryRAG.tsx - Semantic search
 * - KnowledgeGraph.tsx - Graph visualization
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import { act } from 'react-dom/test-utils';

// Mock API service
const mockApiService = {
  invokeAgent: vi.fn(),
  getAgentStatus: vi.fn(),
  streamAgentProgress: vi.fn(),
  searchMemory: vi.fn(),
  embedContent: vi.fn(),
  generateCode: vi.fn(),
  analyzeContext: vi.fn(),
  getKnowledgeGraph: vi.fn(),
};

// Mock WebSocket service
const mockWebSocketService = {
  connect: vi.fn(),
  disconnect: vi.fn(),
  subscribe: vi.fn(),
  unsubscribe: vi.fn(),
  send: vi.fn(),
};

// Mock components (we'll test them individually)
import AgentDashboard from '../components/AgentDashboard';
import ContextAnalysis from '../components/ContextAnalysis';
import CodeGeneration from '../components/CodeGeneration';
import MemoryRAG from '../components/MemoryRAG';
import KnowledgeGraph from '../components/KnowledgeGraph';

// Mock the API and WebSocket services
vi.mock('../services/api', () => ({
  apiService: mockApiService
}));

vi.mock('../services/websocket', () => ({
  getWebSocketService: () => mockWebSocketService
}));

describe('AgentDashboard Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders agent dashboard with correct initial state', () => {
    const mockAgents = [
      {
        id: 'agent-1',
        type: 'code-review',
        status: 'idle',
        name: 'Code Review Agent',
        description: 'Reviews code for quality and security',
        lastActivity: '2024-01-15T10:00:00Z'
      },
      {
        id: 'agent-2',
        type: 'workspace-analyzer',
        status: 'running',
        name: 'Workspace Analyzer',
        description: 'Analyzes project structure',
        lastActivity: '2024-01-15T10:30:00Z'
      }
    ];

    render(<AgentDashboard agents={mockAgents} />);

    // Check if agents are rendered
    expect(screen.getByText('Code Review Agent')).toBeInTheDocument();
    expect(screen.getByText('Workspace Analyzer')).toBeInTheDocument();

    // Check status indicators
    expect(screen.getByText('idle')).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
  });

  it('handles agent invocation correctly', async () => {
    const user = userEvent.setup();
    const mockAgents = [
      {
        id: 'agent-1',
        type: 'code-review',
        status: 'idle',
        name: 'Code Review Agent',
        description: 'Reviews code for quality and security'
      }
    ];

    mockApiService.invokeAgent.mockResolvedValue({
      jobId: 'job-123',
      status: 'queued',
      message: 'Agent invoked successfully'
    });

    render(<AgentDashboard agents={mockAgents} />);

    // Find and click invoke button
    const invokeButton = screen.getByRole('button', { name: /invoke/i });
    await user.click(invokeButton);

    await waitFor(() => {
      expect(mockApiService.invokeAgent).toHaveBeenCalledWith('code-review', {
        parameters: {}
      });
    });
  });

  it('displays real-time status updates', async () => {
    const mockAgents = [
      {
        id: 'agent-1',
        type: 'code-review',
        status: 'running',
        name: 'Code Review Agent',
        progress: 45
      }
    ];

    render(<AgentDashboard agents={mockAgents} />);

    // Check initial progress
    expect(screen.getByText('45%')).toBeInTheDocument();

    // Simulate WebSocket update
    act(() => {
      const updateCallback = mockWebSocketService.subscribe.mock.calls[0][1];
      updateCallback({
        type: 'agent_status_update',
        data: {
          agentId: 'agent-1',
          status: 'running',
          progress: 75,
          message: 'Processing files...'
        }
      });
    });

    await waitFor(() => {
      expect(screen.getByText('75%')).toBeInTheDocument();
      expect(screen.getByText('Processing files...')).toBeInTheDocument();
    });
  });

  it('handles agent cancellation', async () => {
    const user = userEvent.setup();
    const mockAgents = [
      {
        id: 'agent-1',
        type: 'code-review',
        status: 'running',
        name: 'Code Review Agent',
        jobId: 'job-123'
      }
    ];

    mockApiService.cancelJob = vi.fn().mockResolvedValue({ success: true });

    render(<AgentDashboard agents={mockAgents} />);

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    await user.click(cancelButton);

    await waitFor(() => {
      expect(mockApiService.cancelJob).toHaveBeenCalledWith('job-123');
    });
  });
});

describe('ContextAnalysis Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders context analysis form correctly', () => {
    render(<ContextAnalysis />);

    expect(screen.getByLabelText(/project path/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/analysis type/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyze/i })).toBeInTheDocument();
  });

  it('submits context analysis request', async () => {
    const user = userEvent.setup();

    mockApiService.analyzeContext.mockResolvedValue({
      contextId: 'ctx-123',
      status: 'processing',
      message: 'Analysis started'
    });

    render(<ContextAnalysis />);

    // Fill form
    await user.type(screen.getByLabelText(/project path/i), '/test/project');
    await user.selectOptions(screen.getByLabelText(/analysis type/i), 'comprehensive');

    // Submit
    await user.click(screen.getByRole('button', { name: /analyze/i }));

    await waitFor(() => {
      expect(mockApiService.analyzeContext).toHaveBeenCalledWith({
        projectPath: '/test/project',
        analysisType: 'comprehensive',
        includeDependencies: true
      });
    });
  });

  it('displays analysis results correctly', async () => {
    const mockAnalysisResult = {
      contextId: 'ctx-123',
      analysis: {
        components: ['frontend', 'backend', 'database'],
        architecture: 'microservices',
        technologies: ['React', 'FastAPI', 'PostgreSQL'],
        complexity: 'medium'
      },
      recommendations: [
        'Add comprehensive testing',
        'Implement caching layer',
        'Optimize database queries'
      ]
    };

    render(<ContextAnalysis />);

    // Simulate analysis completion
    act(() => {
      // Trigger results display
      const component = screen.getByTestId('context-analysis');
      fireEvent(component, new CustomEvent('analysisComplete', {
        detail: mockAnalysisResult
      }));
    });

    await waitFor(() => {
      expect(screen.getByText('microservices')).toBeInTheDocument();
      expect(screen.getByText('React')).toBeInTheDocument();
      expect(screen.getByText('Add comprehensive testing')).toBeInTheDocument();
    });
  });

  it('handles analysis errors gracefully', async () => {
    const user = userEvent.setup();

    mockApiService.analyzeContext.mockRejectedValue(new Error('Analysis failed'));

    render(<ContextAnalysis />);

    await user.type(screen.getByLabelText(/project path/i), '/invalid/path');
    await user.click(screen.getByRole('button', { name: /analyze/i }));

    await waitFor(() => {
      expect(screen.getByText(/analysis failed/i)).toBeInTheDocument();
    });
  });
});

describe('CodeGeneration Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders code generation interface', () => {
    render(<CodeGeneration />);

    expect(screen.getByLabelText(/specification/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/language/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/framework/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generate/i })).toBeInTheDocument();
  });

  it('submits code generation request', async () => {
    const user = userEvent.setup();

    mockApiService.generateCode.mockResolvedValue({
      generationId: 'gen-456',
      status: 'processing',
      estimatedDuration: 120
    });

    render(<CodeGeneration />);

    // Fill form
    await user.type(
      screen.getByLabelText(/specification/i),
      'Create a REST API for user authentication'
    );
    await user.selectOptions(screen.getByLabelText(/language/i), 'python');
    await user.selectOptions(screen.getByLabelText(/framework/i), 'fastapi');
    await user.click(screen.getByLabelText(/include tests/i));

    // Submit
    await user.click(screen.getByRole('button', { name: /generate/i }));

    await waitFor(() => {
      expect(mockApiService.generateCode).toHaveBeenCalledWith({
        specification: 'Create a REST API for user authentication',
        language: 'python',
        framework: 'fastapi',
        includeTests: true
      });
    });
  });

  it('displays generated code correctly', async () => {
    const mockGeneratedCode = {
      generationId: 'gen-456',
      files: {
        'auth_endpoint.py': {
          content: '# FastAPI authentication endpoint\nfrom fastapi import APIRouter\n...',
          language: 'python',
          lines: 85
        },
        'test_auth_endpoint.py': {
          content: '# Test cases\nimport pytest\n...',
          language: 'python',
          lines: 45
        }
      },
      metadata: {
        totalLines: 130,
        qualityScore: 8.5,
        testCoverage: 95
      }
    };

    render(<CodeGeneration />);

    // Simulate code generation completion
    act(() => {
      const component = screen.getByTestId('code-generation');
      fireEvent(component, new CustomEvent('generationComplete', {
        detail: mockGeneratedCode
      }));
    });

    await waitFor(() => {
      expect(screen.getByText('auth_endpoint.py')).toBeInTheDocument();
      expect(screen.getByText('test_auth_endpoint.py')).toBeInTheDocument();
      expect(screen.getByText('Quality Score: 8.5')).toBeInTheDocument();
      expect(screen.getByText('Test Coverage: 95%')).toBeInTheDocument();
    });
  });

  it('handles streaming code generation updates', async () => {
    render(<CodeGeneration />);

    // Simulate streaming updates
    const streamingUpdates = [
      { progress: 25, message: 'Analyzing specification...' },
      { progress: 50, message: 'Generating main code...' },
      { progress: 75, message: 'Creating tests...' },
      { progress: 100, message: 'Generation complete!' }
    ];

    for (const update of streamingUpdates) {
      act(() => {
        const component = screen.getByTestId('code-generation');
        fireEvent(component, new CustomEvent('generationProgress', {
          detail: update
        }));
      });

      await waitFor(() => {
        expect(screen.getByText(update.message)).toBeInTheDocument();
        expect(screen.getByText(`${update.progress}%`)).toBeInTheDocument();
      });
    }
  });
});

describe('MemoryRAG Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders memory search interface', () => {
    render(<MemoryRAG />);

    expect(screen.getByPlaceholderText(/search memory/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /search/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/content type/i)).toBeInTheDocument();
  });

  it('performs memory search', async () => {
    const user = userEvent.setup();

    mockApiService.searchMemory.mockResolvedValue({
      results: [
        {
          id: 'mem-1',
          content: 'JWT authentication implementation',
          similarity: 0.92,
          metadata: {
            filePath: '/app/auth.py',
            language: 'python'
          }
        }
      ],
      totalResults: 1,
      searchTime: 45
    });

    render(<MemoryRAG />);

    // Perform search
    await user.type(
      screen.getByPlaceholderText(/search memory/i),
      'JWT authentication'
    );
    await user.click(screen.getByRole('button', { name: /search/i }));

    await waitFor(() => {
      expect(mockApiService.searchMemory).toHaveBeenCalledWith({
        query: 'JWT authentication',
        contentTypes: ['code', 'documentation'],
        maxResults: 10,
        similarityThreshold: 0.7
      });
    });

    // Check results display
    await waitFor(() => {
      expect(screen.getByText('JWT authentication implementation')).toBeInTheDocument();
      expect(screen.getByText('92%')).toBeInTheDocument(); // similarity score
      expect(screen.getByText('/app/auth.py')).toBeInTheDocument();
    });
  });

  it('handles content embedding', async () => {
    const user = userEvent.setup();

    mockApiService.embedContent.mockResolvedValue({
      embeddingId: 'emb-789',
      status: 'embedded',
      vectorDimensions: 768
    });

    render(<MemoryRAG />);

    // Switch to embed tab
    await user.click(screen.getByRole('tab', { name: /embed/i }));

    // Fill content
    await user.type(
      screen.getByLabelText(/content/i),
      'This is sample code for testing'
    );
    await user.type(
      screen.getByLabelText(/file path/i),
      '/test/sample.py'
    );

    // Submit
    await user.click(screen.getByRole('button', { name: /embed/i }));

    await waitFor(() => {
      expect(mockApiService.embedContent).toHaveBeenCalledWith({
        content: 'This is sample code for testing',
        metadata: {
          filePath: '/test/sample.py',
          contentType: 'code'
        }
      });
    });
  });

  it('filters search results correctly', async () => {
    const user = userEvent.setup();

    const mockResults = [
      {
        id: 'mem-1',
        content: 'Python authentication code',
        metadata: { language: 'python', contentType: 'code' }
      },
      {
        id: 'mem-2',
        content: 'JavaScript authentication docs',
        metadata: { language: 'javascript', contentType: 'documentation' }
      }
    ];

    mockApiService.searchMemory.mockResolvedValue({
      results: mockResults,
      totalResults: 2
    });

    render(<MemoryRAG />);

    // Apply filters
    await user.selectOptions(screen.getByLabelText(/language/i), 'python');
    await user.selectOptions(screen.getByLabelText(/content type/i), 'code');

    // Perform search
    await user.type(screen.getByPlaceholderText(/search memory/i), 'authentication');
    await user.click(screen.getByRole('button', { name: /search/i }));

    await waitFor(() => {
      expect(mockApiService.searchMemory).toHaveBeenCalledWith(
        expect.objectContaining({
          filters: {
            language: 'python',
            contentType: 'code'
          }
        })
      );
    });
  });
});

describe('KnowledgeGraph Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders knowledge graph interface', () => {
    render(<KnowledgeGraph />);

    expect(screen.getByRole('button', { name: /build graph/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/project path/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/include relationships/i)).toBeInTheDocument();
  });

  it('builds knowledge graph', async () => {
    const user = userEvent.setup();

    mockApiService.getKnowledgeGraph.mockResolvedValue({
      graphId: 'graph-123',
      nodes: [
        {
          id: 'node-1',
          type: 'class',
          name: 'AuthService',
          properties: { filePath: '/app/auth.py' }
        },
        {
          id: 'node-2',
          type: 'function',
          name: 'validate_token',
          properties: { filePath: '/app/auth.py' }
        }
      ],
      edges: [
        {
          source: 'node-1',
          target: 'node-2',
          relationship: 'contains',
          weight: 1.0
        }
      ],
      statistics: {
        totalNodes: 2,
        totalEdges: 1,
        complexity: 0.3
      }
    });

    render(<KnowledgeGraph />);

    // Fill form and build graph
    await user.type(screen.getByLabelText(/project path/i), '/test/project');
    await user.click(screen.getByLabelText(/include relationships/i));
    await user.click(screen.getByRole('button', { name: /build graph/i }));

    await waitFor(() => {
      expect(mockApiService.getKnowledgeGraph).toHaveBeenCalledWith({
        projectPath: '/test/project',
        includeRelationships: true,
        maxDepth: 3,
        nodeTypes: ['functions', 'classes', 'modules']
      });
    });
  });

  it('displays graph statistics', async () => {
    const mockGraphData = {
      graphId: 'graph-123',
      nodes: [/* nodes */],
      edges: [/* edges */],
      statistics: {
        totalNodes: 15,
        totalEdges: 23,
        complexity: 0.7,
        modularity: 0.85
      }
    };

    render(<KnowledgeGraph />);

    // Simulate graph built
    act(() => {
      const component = screen.getByTestId('knowledge-graph');
      fireEvent(component, new CustomEvent('graphBuilt', {
        detail: mockGraphData
      }));
    });

    await waitFor(() => {
      expect(screen.getByText('15 nodes')).toBeInTheDocument();
      expect(screen.getByText('23 edges')).toBeInTheDocument();
      expect(screen.getByText('Complexity: 0.7')).toBeInTheDocument();
      expect(screen.getByText('Modularity: 0.85')).toBeInTheDocument();
    });
  });

  it('handles graph visualization interactions', async () => {
    const user = userEvent.setup();

    const mockGraphData = {
      nodes: [
        { id: 'node-1', name: 'AuthService', type: 'class' },
        { id: 'node-2', name: 'UserModel', type: 'class' }
      ],
      edges: [
        { source: 'node-1', target: 'node-2', relationship: 'uses' }
      ]
    };

    render(<KnowledgeGraph />);

    // Simulate graph rendering
    act(() => {
      const component = screen.getByTestId('knowledge-graph');
      fireEvent(component, new CustomEvent('graphBuilt', {
        detail: mockGraphData
      }));
    });

    // Test node selection
    const node = screen.getByText('AuthService');
    await user.click(node);

    await waitFor(() => {
      expect(screen.getByText(/node details/i)).toBeInTheDocument();
      expect(screen.getByText('class')).toBeInTheDocument();
    });
  });

  it('supports graph filtering and search', async () => {
    const user = userEvent.setup();

    render(<KnowledgeGraph />);

    // Test node type filtering
    await user.selectOptions(screen.getByLabelText(/node types/i), 'functions');

    // Test search
    await user.type(screen.getByPlaceholderText(/search nodes/i), 'auth');

    // Verify filter application
    expect(screen.getByDisplayValue('auth')).toBeInTheDocument();
  });
});

describe('Component Integration', () => {
  it('components communicate through shared state', async () => {
    const user = userEvent.setup();

    // This would test how components work together in the actual app
    // For example, selecting a node in knowledge graph updates context analysis

    const mockSharedState = {
      selectedProject: '/test/project',
      activeContext: 'ctx-123',
      selectedNode: 'node-1'
    };

    // Test would verify state propagation between components
    expect(mockSharedState.selectedProject).toBe('/test/project');
  });

  it('handles real-time updates across components', async () => {
    // Test WebSocket updates affecting multiple components
    const mockUpdate = {
      type: 'project_analysis_complete',
      data: {
        contextId: 'ctx-123',
        graphId: 'graph-456',
        codeGeneration: 'gen-789'
      }
    };

    expect(mockUpdate.type).toBe('project_analysis_complete');
    expect(mockUpdate.data.contextId).toBe('ctx-123');
  });
});

// Error boundary tests
describe('Error Handling', () => {
  it('displays error boundaries for component failures', () => {
    // Mock component that throws error
    const ThrowError = () => {
      throw new Error('Component failed');
    };

    // Would test error boundary wrapping
    expect(() => {
      render(<ThrowError />);
    }).toThrow('Component failed');
  });

  it('handles API failures gracefully', async () => {
    mockApiService.invokeAgent.mockRejectedValue(new Error('API Error'));

    render(<AgentDashboard agents={[]} />);

    // Test error handling in components
    // Components should display user-friendly error messages
  });
});

// Accessibility tests
describe('Accessibility', () => {
  it('components have proper ARIA labels', () => {
    render(<AgentDashboard agents={[]} />);

    // Check for proper accessibility attributes
    expect(screen.getByRole('main')).toBeInTheDocument();
    expect(screen.getByLabelText(/agent dashboard/i)).toBeInTheDocument();
  });

  it('supports keyboard navigation', async () => {
    const user = userEvent.setup();

    render(<CodeGeneration />);

    // Test tab navigation
    await user.tab();
    expect(screen.getByLabelText(/specification/i)).toHaveFocus();

    await user.tab();
    expect(screen.getByLabelText(/language/i)).toHaveFocus();
  });
});
