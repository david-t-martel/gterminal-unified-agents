package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/charmbracelet/log"
	"github.com/gorilla/websocket"

	"go-react-terminal/internal/models"
)

// ReactClient handles communication with the Python Enhanced ReAct backend
type ReactClient struct {
	baseURL    string
	httpClient *http.Client
	logger     *log.Logger
	wsConn     *websocket.Conn
}

// NewReactClient creates a new React client
func NewReactClient(baseURL string, logger *log.Logger) (*ReactClient, error) {
	// Validate URL
	if _, err := url.Parse(baseURL); err != nil {
		return nil, fmt.Errorf("invalid base URL: %w", err)
	}

	// Create HTTP client with reasonable timeouts
	httpClient := &http.Client{
		Timeout: 30 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        10,
			IdleConnTimeout:     30 * time.Second,
			DisableCompression:  false,
			MaxIdleConnsPerHost: 3,
		},
	}

	return &ReactClient{
		baseURL:    baseURL,
		httpClient: httpClient,
		logger:     logger,
	}, nil
}

// TestConnection verifies connectivity to the backend
func (c *ReactClient) TestConnection() error {
	url := fmt.Sprintf("%s/api/health", c.baseURL)
	
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return fmt.Errorf("connection test failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("backend returned status %d: %s", resp.StatusCode, string(body))
	}

	c.logger.Info("Successfully connected to ReAct backend", "url", c.baseURL)
	return nil
}

// GetSystemMetrics retrieves comprehensive system metrics
func (c *ReactClient) GetSystemMetrics() (*models.SystemMetrics, error) {
	url := fmt.Sprintf("%s/api/metrics", c.baseURL)
	
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get metrics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics request failed with status %d", resp.StatusCode)
	}

	var metrics models.SystemMetrics
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		return nil, fmt.Errorf("failed to decode metrics response: %w", err)
	}

	return &metrics, nil
}

// ProcessTask executes a ReAct task and returns the result
func (c *ReactClient) ProcessTask(task models.ReactTask, progressCallback func(models.TaskProgress)) (*models.TaskResult, error) {
	// Prepare request payload
	payload := map[string]interface{}{
		"description":      task.Description,
		"priority":         task.Priority,
		"privacy_level":    task.PrivacyLevel,
		"required_agents":  task.RequiredAgents,
		"context":          task.Context,
		"constraints":      task.Constraints,
		"success_criteria": task.SuccessCriteria,
		"estimated_duration": task.EstimatedDuration,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal task payload: %w", err)
	}

	// Send HTTP request to process task
	url := fmt.Sprintf("%s/api/tasks", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("task request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("task processing failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result models.TaskResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode task result: %w", err)
	}

	return &result, nil
}

// ProcessTaskWithProgress executes a task with real-time progress updates via WebSocket
func (c *ReactClient) ProcessTaskWithProgress(task models.ReactTask, progressCallback func(models.TaskProgress)) (*models.TaskResult, error) {
	// First, establish WebSocket connection for progress updates
	if err := c.connectWebSocket(); err != nil {
		c.logger.Warn("WebSocket connection failed, falling back to HTTP only", "error", err)
		return c.ProcessTask(task, progressCallback)
	}
	defer c.closeWebSocket()

	// Start listening for progress updates in a goroutine
	progressDone := make(chan struct{})
	go c.listenForProgress(progressCallback, progressDone)

	// Execute the task via HTTP
	result, err := c.ProcessTask(task, progressCallback)
	
	// Stop progress monitoring
	close(progressDone)
	
	return result, err
}

// connectWebSocket establishes a WebSocket connection for real-time updates
func (c *ReactClient) connectWebSocket() error {
	wsURL := fmt.Sprintf("ws://%s/ws/go_client", c.parseHost())
	
	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	conn, _, err := dialer.Dial(wsURL, nil)
	if err != nil {
		return fmt.Errorf("WebSocket dial failed: %w", err)
	}

	c.wsConn = conn
	return nil
}

// listenForProgress listens for progress updates via WebSocket
func (c *ReactClient) listenForProgress(callback func(models.TaskProgress), done chan struct{}) {
	if c.wsConn == nil {
		return
	}

	for {
		select {
		case <-done:
			return
		default:
			var progress models.TaskProgress
			if err := c.wsConn.ReadJSON(&progress); err != nil {
				c.logger.Warn("Failed to read progress update", "error", err)
				return
			}
			if callback != nil {
				callback(progress)
			}
		}
	}
}

// closeWebSocket closes the WebSocket connection
func (c *ReactClient) closeWebSocket() {
	if c.wsConn != nil {
		c.wsConn.Close()
		c.wsConn = nil
	}
}

// GetTaskHistory retrieves recent task execution history
func (c *ReactClient) GetTaskHistory(limit int) ([]models.TaskResult, error) {
	url := fmt.Sprintf("%s/api/tasks/history?limit=%d", c.baseURL, limit)
	
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get task history: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("task history request failed with status %d", resp.StatusCode)
	}

	var history []models.TaskResult
	if err := json.NewDecoder(resp.Body).Decode(&history); err != nil {
		return nil, fmt.Errorf("failed to decode history response: %w", err)
	}

	return history, nil
}

// GetSessionInfo retrieves information about active sessions
func (c *ReactClient) GetSessionInfo() (*models.SessionInfo, error) {
	url := fmt.Sprintf("%s/api/sessions", c.baseURL)
	
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to get session info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("session info request failed with status %d", resp.StatusCode)
	}

	var sessionInfo models.SessionInfo
	if err := json.NewDecoder(resp.Body).Decode(&sessionInfo); err != nil {
		return nil, fmt.Errorf("failed to decode session response: %w", err)
	}

	return &sessionInfo, nil
}

// ExecuteCommand sends a command to the backend for processing
func (c *ReactClient) ExecuteCommand(command string) (*models.CommandResult, error) {
	payload := map[string]string{
		"command": command,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal command payload: %w", err)
	}

	url := fmt.Sprintf("%s/api/commands", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("command request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("command execution failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result models.CommandResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode command result: %w", err)
	}

	return &result, nil
}

// Shutdown gracefully closes the client
func (c *ReactClient) Shutdown() error {
	c.closeWebSocket()
	
	// Close idle connections
	if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
	
	c.logger.Info("ReAct client shutdown completed")
	return nil
}

// parseHost extracts host from base URL for WebSocket connections
func (c *ReactClient) parseHost() string {
	u, err := url.Parse(c.baseURL)
	if err != nil {
		return "localhost:8080" // fallback
	}
	return u.Host
}

// HealthCheck performs a comprehensive health check
func (c *ReactClient) HealthCheck() (*models.HealthStatus, error) {
	url := fmt.Sprintf("%s/api/health/detailed", c.baseURL)
	
	resp, err := c.httpClient.Get(url)
	if err != nil {
		return &models.HealthStatus{
			Status:    "unhealthy",
			Message:   fmt.Sprintf("Connection failed: %v", err),
			Timestamp: time.Now(),
		}, err
	}
	defer resp.Body.Close()

	var health models.HealthStatus
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return &models.HealthStatus{
			Status:    "unknown",
			Message:   fmt.Sprintf("Failed to decode health response: %v", err),
			Timestamp: time.Now(),
		}, err
	}

	return &health, nil
}