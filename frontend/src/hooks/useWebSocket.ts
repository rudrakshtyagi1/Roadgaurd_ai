import { useState, useEffect, useRef, useCallback } from 'react';
import { WSMessage } from '../types';

interface UseWebSocketReturn {
  data: WSMessage | null;
  connected: boolean;
  reconnecting: boolean;
}

const WS_URL = 'ws://localhost:8000/ws';
const MAX_RETRIES = 5;

export function useWebSocket(): UseWebSocketReturn {
  const [data, setData] = useState<WSMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const ws = useRef<WebSocket | null>(null);
  const retries = useRef(0);
  const timeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    const socket = new WebSocket(WS_URL);
    ws.current = socket;

    socket.onopen = () => {
      setConnected(true);
      setReconnecting(false);
      retries.current = 0;
    };

    socket.onmessage = (event) => {
      try {
        const parsed: WSMessage = JSON.parse(event.data);
        setData(parsed);
      } catch { /* ignore parse errors */ }
    };

    socket.onclose = () => {
      setConnected(false);
      if (retries.current < MAX_RETRIES) {
        setReconnecting(true);
        const delay = Math.min(1000 * 2 ** retries.current, 16000);
        retries.current++;
        timeout.current = setTimeout(connect, delay);
      } else {
        setReconnecting(false);
      }
    };

    socket.onerror = () => socket.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (timeout.current) clearTimeout(timeout.current);
      ws.current?.close();
    };
  }, [connect]);

  return { data, connected, reconnecting };
}
