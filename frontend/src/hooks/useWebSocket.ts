import { useEffect, useRef } from 'react';
import { getWebSocketService, initializeWebSocket } from '@/services/websocket';

export const useWebSocket = (url?: string) => {
  const wsRef = useRef<any>(null);

  useEffect(() => {
    if (url) {
      try {
        wsRef.current = initializeWebSocket(url);
        wsRef.current.connect().catch(console.error);

        return () => {
          if (wsRef.current) {
            wsRef.current.disconnect();
          }
        };
      } catch (error) {
        console.warn('WebSocket initialization failed:', error);
      }
    }
  }, [url]);

  return wsRef.current;
};

export const useGlobalWebSocket = () => {
  const wsRef = useRef<any>(null);

  useEffect(() => {
    try {
      const wsBaseUrl = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsBaseUrl}//${window.location.host}/api/v1/ws`;

      wsRef.current = getWebSocketService(wsUrl);
      wsRef.current.connect().catch(console.error);

      return () => {
        if (wsRef.current) {
          wsRef.current.disconnect();
        }
      };
    } catch (error) {
      console.warn('Global WebSocket not available:', error);
    }
  }, []);

  return wsRef.current;
};
