import { AgentType, WebSocketMessage, AgentStatusEvent, GenerationProgressEvent } from '@/types/api';

export type WebSocketEventHandler = (data: any) => void;

export interface WebSocketOptions {
  reconnectAttempts?: number;
  reconnectDelay?: number;
  heartbeatInterval?: number;
}

export class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private options: WebSocketOptions;
  private reconnectCount = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private eventHandlers: Map<string, WebSocketEventHandler[]> = new Map();
  private isConnecting = false;
  private isManuallyDisconnected = false;

  constructor(url: string, options: WebSocketOptions = {}) {
    this.url = url;
    this.options = {
      reconnectAttempts: 10,
      reconnectDelay: 1000,
      heartbeatInterval: 30000,
      ...options
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        reject(new Error('Already attempting to connect'));
        return;
      }

      this.isConnecting = true;
      this.isManuallyDisconnected = false;

      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectCount = 0;
          this.startHeartbeat();
          this.emit('connected', {});
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.stopHeartbeat();
          this.emit('disconnected', { code: event.code, reason: event.reason });

          if (!this.isManuallyDisconnected) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          this.emit('error', error);
          reject(error);
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect(): void {
    this.isManuallyDisconnected = true;
    this.clearReconnectTimer();
    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close(1000, 'Manual disconnect');
      this.ws = null;
    }
  }

  send(data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, cannot send data');
    }
  }

  // Agent-specific methods
  subscribeToAgent(agentType: AgentType, agentId?: string): void {
    this.send({
      type: 'subscribe',
      data: {
        topic: 'agent_status',
        agentType,
        agentId
      }
    });
  }

  unsubscribeFromAgent(agentType: AgentType, agentId?: string): void {
    this.send({
      type: 'unsubscribe',
      data: {
        topic: 'agent_status',
        agentType,
        agentId
      }
    });
  }

  subscribeToGeneration(requestId: string): void {
    this.send({
      type: 'subscribe',
      data: {
        topic: 'generation_progress',
        requestId
      }
    });
  }

  unsubscribeFromGeneration(requestId: string): void {
    this.send({
      type: 'unsubscribe',
      data: {
        topic: 'generation_progress',
        requestId
      }
    });
  }

  // Event handlers
  on(event: string, handler: WebSocketEventHandler): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  off(event: string, handler?: WebSocketEventHandler): void {
    if (!handler) {
      this.eventHandlers.delete(event);
      return;
    }

    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error('Error in WebSocket event handler:', error);
        }
      });
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'agent_status':
        this.emit('agent_status', message.data as AgentStatusEvent);
        break;
      case 'generation_progress':
        this.emit('generation_progress', message.data as GenerationProgressEvent);
        break;
      case 'error':
        this.emit('error', message.data);
        break;
      case 'completion':
        this.emit('completion', message.data);
        break;
      default:
        this.emit('message', message);
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectCount >= this.options.reconnectAttempts!) {
      console.error('Max reconnection attempts reached');
      this.emit('max_reconnect_attempts', {});
      return;
    }

    const delay = this.options.reconnectDelay! * Math.pow(2, this.reconnectCount);
    console.log(`Scheduling reconnect in ${delay}ms (attempt ${this.reconnectCount + 1})`);

    this.reconnectTimer = setTimeout(() => {
      this.reconnectCount++;
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, this.options.heartbeatInterval!);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get connectionState(): string {
    if (!this.ws) return 'disconnected';
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'connecting';
      case WebSocket.OPEN: return 'connected';
      case WebSocket.CLOSING: return 'closing';
      case WebSocket.CLOSED: return 'disconnected';
      default: return 'unknown';
    }
  }
}

// Singleton instance
let webSocketService: WebSocketService | null = null;

export const getWebSocketService = (url?: string): WebSocketService => {
  if (!webSocketService && url) {
    webSocketService = new WebSocketService(url);
  }
  if (!webSocketService) {
    throw new Error('WebSocket service not initialized. Provide URL on first call.');
  }
  return webSocketService;
};

export const initializeWebSocket = (url: string, options?: WebSocketOptions): WebSocketService => {
  if (webSocketService) {
    webSocketService.disconnect();
  }
  webSocketService = new WebSocketService(url, options);
  return webSocketService;
};
