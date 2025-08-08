package models

import (
	"fmt"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// ReactTask represents a task to be processed by the Enhanced ReAct engine
type ReactTask struct {
	Description       string                 `json:"description"`
	Priority          string                 `json:"priority"`           // "high", "medium", "low"
	PrivacyLevel      string                 `json:"privacy_level"`      // "standard", "sensitive", "private"
	RequiredAgents    []string               `json:"required_agents"`    // Agent roles required
	Context           map[string]interface{} `json:"context"`            // Additional context
	Constraints       []string               `json:"constraints"`        // Task constraints
	SuccessCriteria   []string               `json:"success_criteria"`   // Success measurements
	EstimatedDuration int                    `json:"estimated_duration"` // Seconds
	Timestamp         time.Time              `json:"timestamp"`
}

// TaskResult represents the result of processing a ReAct task
type TaskResult struct {
	TaskID          string                 `json:"task_id"`
	Description     string                 `json:"description"`
	Success         bool                   `json:"success"`
	Result          string                 `json:"result"`
	ExecutionTime   float64                `json:"execution_time"`
	Iterations      int                    `json:"iterations"`
	ActionsCount    int                    `json:"actions_count"`
	UsedLocalLLM    bool                   `json:"used_local_llm"`
	PerformanceMetrics map[string]interface{} `json:"performance_metrics"`
	Timestamp       time.Time              `json:"timestamp"`
	Error           string                 `json:"error,omitempty"`
}

// TaskProgress represents real-time progress updates during task execution
type TaskProgress struct {
	TaskID       string `json:"task_id"`
	Iteration    int    `json:"iteration"`
	Status       string `json:"status"`
	ActionsCount int    `json:"actions_count"`
	LatestStep   string `json:"latest_step"`
	Timestamp    time.Time `json:"timestamp"`
}

// SystemMetrics represents comprehensive system metrics from the backend
type SystemMetrics struct {
	RustExtensions       bool                    `json:"rust_extensions"`
	LocalLLMAvailable    bool                    `json:"local_llm_available"`
	WebFetchAvailable    bool                    `json:"web_fetch_available"`
	OrchestratorMetrics  *OrchestratorMetrics    `json:"orchestrator_metrics"`
	MessageQueueStats    *MessageQueueStats      `json:"message_queue_stats"`
	SystemInfo           *SystemInfo             `json:"system_info"`
	Timestamp            time.Time               `json:"timestamp"`
}

// OrchestratorMetrics represents metrics from the ReAct orchestrator
type OrchestratorMetrics struct {
	TasksCompleted    int     `json:"tasks_completed"`
	TasksFailed       int     `json:"tasks_failed"`
	AverageTaskTime   float64 `json:"average_task_time"`
	LLMCalls          int     `json:"llm_calls"`
	LocalLLMCalls     int     `json:"local_llm_calls"`
	CacheHits         int     `json:"cache_hits"`
	ToolsExecuted     int     `json:"tools_executed"`
}

// MessageQueueStats represents message queue performance metrics
type MessageQueueStats struct {
	Sent         int `json:"sent"`
	Processed    int `json:"processed"`
	ActiveQueues int `json:"active_queues"`
	Failed       int `json:"failed"`
}

// SystemInfo represents system information from the backend
type SystemInfo struct {
	PythonVersion string `json:"python_version"`
	Platform      string `json:"platform"`
	MemoryUsage   int64  `json:"memory_usage"`
	CPUUsage      float64 `json:"cpu_usage"`
	Uptime        int64  `json:"uptime"`
}

// SessionInfo represents information about active sessions
type SessionInfo struct {
	SessionID     string              `json:"session_id"`
	ClientType    string              `json:"client_type"`
	StartTime     time.Time           `json:"start_time"`
	CommandCount  int                 `json:"command_count"`
	TaskCount     int                 `json:"task_count"`
	LastActivity  time.Time           `json:"last_activity"`
	Capabilities  []string            `json:"capabilities"`
	Configuration map[string]interface{} `json:"configuration"`
}

// CommandResult represents the result of executing a command
type CommandResult struct {
	Command       string                 `json:"command"`
	Success       bool                   `json:"success"`
	Output        string                 `json:"output"`
	Error         string                 `json:"error,omitempty"`
	ExecutionTime float64                `json:"execution_time"`
	Metadata      map[string]interface{} `json:"metadata"`
	Timestamp     time.Time              `json:"timestamp"`
}

// HealthStatus represents the health status of the backend
type HealthStatus struct {
	Status       string                 `json:"status"`      // "healthy", "unhealthy", "degraded"
	Message      string                 `json:"message"`
	Components   map[string]interface{} `json:"components"`  // Status of individual components
	Version      string                 `json:"version"`
	Uptime       int64                  `json:"uptime"`
	Timestamp    time.Time              `json:"timestamp"`
}

// Bubble Tea Messages for handling async operations

// StatusMsg represents a status update message
type StatusMsg struct {
	Message string
}

// ErrorMsg represents an error message
type ErrorMsg struct {
	Err error
}

// TaskProgressMsg represents a task progress update
type TaskProgressMsg struct {
	Progress TaskProgress
}

// TaskCompletedMsg represents task completion
type TaskCompletedMsg struct {
	Result TaskResult
}

// SystemMetricsMsg represents system metrics update
type SystemMetricsMsg struct {
	Metrics *SystemMetrics
}

// ConnectionStatusMsg represents connection status changes
type ConnectionStatusMsg struct {
	Connected bool
	Message   string
}

// CommandExecutedMsg represents command execution completion
type CommandExecutedMsg struct {
	Result CommandResult
}

// Session management types

// SessionState represents the state of a client session
type SessionState struct {
	ID              string                 `json:"id"`
	StartTime       time.Time              `json:"start_time"`
	LastActivity    time.Time              `json:"last_activity"`
	CommandHistory  []string               `json:"command_history"`
	TaskHistory     []TaskResult           `json:"task_history"`
	CurrentTask     *ReactTask             `json:"current_task,omitempty"`
	Configuration   map[string]interface{} `json:"configuration"`
	ClientInfo      ClientInfo             `json:"client_info"`
}

// ClientInfo represents information about the client
type ClientInfo struct {
	Type      string `json:"type"`      // "go-bubbletea", "python-textual", "web"
	Version   string `json:"version"`
	UserAgent string `json:"user_agent"`
	Platform  string `json:"platform"`
}

// Agent types and roles

// AgentRole represents different agent roles in the system
type AgentRole int

const (
	AgentCoordinator AgentRole = iota
	AgentArchitect
	AgentReviewer
	AgentAnalyzer
	AgentExecutor
	AgentResearcher
)

// String returns the string representation of an AgentRole
func (a AgentRole) String() string {
	switch a {
	case AgentCoordinator:
		return "coordinator"
	case AgentArchitect:
		return "architect"
	case AgentReviewer:
		return "reviewer"
	case AgentAnalyzer:
		return "analyzer"
	case AgentExecutor:
		return "executor"
	case AgentResearcher:
		return "researcher"
	default:
		return "unknown"
	}
}

// Message types for multi-agent communication

// AgentMessage represents a message between agents
type AgentMessage struct {
	ID          string                 `json:"id"`
	FromAgent   AgentRole              `json:"from_agent"`
	ToAgent     AgentRole              `json:"to_agent"`
	MessageType string                 `json:"message_type"`
	Content     string                 `json:"content"`
	Metadata    map[string]interface{} `json:"metadata"`
	Timestamp   time.Time              `json:"timestamp"`
	Priority    int                    `json:"priority"`
}

// TaskPriority represents task priority levels
type TaskPriority int

const (
	PriorityLow TaskPriority = iota
	PriorityMedium
	PriorityHigh
	PriorityCritical
)

// String returns the string representation of a TaskPriority
func (p TaskPriority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityMedium:
		return "medium"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "medium"
	}
}

// Configuration types

// ClientConfig represents the configuration for the Go client
type ClientConfig struct {
	BackendURL       string        `json:"backend_url"`
	ConnectTimeout   time.Duration `json:"connect_timeout"`
	RequestTimeout   time.Duration `json:"request_timeout"`
	EnableWebSocket  bool          `json:"enable_websocket"`
	EnableProgress   bool          `json:"enable_progress"`
	LogLevel         string        `json:"log_level"`
	UITheme          string        `json:"ui_theme"`
	AutoRefresh      time.Duration `json:"auto_refresh"`
	MaxHistory       int           `json:"max_history"`
	EnableVimKeys    bool          `json:"enable_vim_keys"`
}

// Default configuration values
var DefaultConfig = ClientConfig{
	BackendURL:       "http://localhost:8080",
	ConnectTimeout:   10 * time.Second,
	RequestTimeout:   30 * time.Second,
	EnableWebSocket:  true,
	EnableProgress:   true,
	LogLevel:         "info",
	UITheme:          "default",
	AutoRefresh:      5 * time.Second,
	MaxHistory:       100,
	EnableVimKeys:    true,
}

// Performance metrics types

// PerformanceMetrics represents performance metrics for the client
type PerformanceMetrics struct {
	StartupTime      time.Duration `json:"startup_time"`
	MemoryUsage      int64         `json:"memory_usage"`
	ConnectionTime   time.Duration `json:"connection_time"`
	LastRequestTime  time.Duration `json:"last_request_time"`
	CommandsExecuted int           `json:"commands_executed"`
	TasksCompleted   int           `json:"tasks_completed"`
	CacheHitRate     float64       `json:"cache_hit_rate"`
	UIUpdateRate     float64       `json:"ui_update_rate"`
}

// WebSocket message types

// WSMessageType represents different WebSocket message types
type WSMessageType string

const (
	WSMessageProgress WSMessageType = "progress"
	WSMessageStatus   WSMessageType = "status"
	WSMessageError    WSMessageType = "error"
	WSMessageCommand  WSMessageType = "command"
	WSMessageMetrics  WSMessageType = "metrics"
)

// WSMessage represents a WebSocket message
type WSMessage struct {
	Type      WSMessageType          `json:"type"`
	SessionID string                 `json:"session_id"`
	Content   map[string]interface{} `json:"content"`
	Timestamp time.Time              `json:"timestamp"`
}

// Error types

// ClientError represents errors that occur in the client
type ClientError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
	Code    int    `json:"code"`
}

func (e ClientError) Error() string {
	return fmt.Sprintf("%s: %s (code: %d)", e.Type, e.Message, e.Code)
}

// Common error types
var (
	ErrConnectionFailed = ClientError{Type: "connection", Message: "Failed to connect to backend", Code: 1001}
	ErrRequestTimeout   = ClientError{Type: "timeout", Message: "Request timed out", Code: 1002}
	ErrInvalidResponse  = ClientError{Type: "response", Message: "Invalid response from backend", Code: 1003}
	ErrTaskFailed       = ClientError{Type: "task", Message: "Task execution failed", Code: 2001}
	ErrCommandFailed    = ClientError{Type: "command", Message: "Command execution failed", Code: 2002}
)

// Utility functions

// NewReactTask creates a new ReactTask with defaults
func NewReactTask(description string) *ReactTask {
	return &ReactTask{
		Description:       description,
		Priority:          PriorityMedium.String(),
		PrivacyLevel:      "standard",
		RequiredAgents:    []string{},
		Context:           make(map[string]interface{}),
		Constraints:       []string{},
		SuccessCriteria:   []string{},
		EstimatedDuration: 0,
		Timestamp:         time.Now(),
	}
}

// NewSessionState creates a new session state
func NewSessionState(clientType string) *SessionState {
	return &SessionState{
		ID:             fmt.Sprintf("go-client-%d", time.Now().Unix()),
		StartTime:      time.Now(),
		LastActivity:   time.Now(),
		CommandHistory: make([]string, 0),
		TaskHistory:    make([]TaskResult, 0),
		Configuration:  make(map[string]interface{}),
		ClientInfo: ClientInfo{
			Type:      clientType,
			Version:   "1.0.0",
			UserAgent: "Go-BubbleTea-ReactClient/1.0.0",
			Platform:  "linux",
		},
	}
}