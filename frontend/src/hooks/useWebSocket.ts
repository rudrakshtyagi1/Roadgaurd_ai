import { useState, useEffect, useCallback } from 'react';
import { WSMessage } from '../types';

const WS_URL = import.meta.env.VITE_WS_URL || (
  import.meta.env.VITE_API_URL
    ? `${import.meta.env.VITE_API_URL.replace(/\/$/, '').replace(/^http/, 'ws')}/ws`
    : import.meta.env.DEV ? 'ws://localhost:8000/ws' : ''
);
const MAX_RETRIES = 5;
const STREAM_TIMEOUT_MS = 15000;

export function useWebSocket() {
  const [data, setData] = useState<WSMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const retry = useCallback(() => setAttempt(value => value + 1), []);

  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | null = null;
    let retries = 0;
    let retryTimer: ReturnType<typeof setTimeout>;
    let streamTimer: ReturnType<typeof setTimeout>;
    setData(null);
    setConnected(false);
    setError(null);
    setReconnecting(false);

    const connect = () => {
      if (disposed) return;
      if (!WS_URL) {
        setError('Live backend is not configured. Set VITE_API_URL or VITE_WS_URL and redeploy.');
        return;
      }
      let url: URL;
      try {
        url = new URL(WS_URL);
        if (!['ws:', 'wss:'].includes(url.protocol)) throw new Error();
        if (window.location.protocol === 'https:' && url.protocol !== 'wss:') {
          setError('This HTTPS dashboard requires a secure wss:// backend connection.');
          return;
        }
      } catch {
        setError('The backend WebSocket URL is invalid. Check VITE_WS_URL.');
        return;
      }
      setReconnecting(true);
      const armTimeout = () => {
        clearTimeout(streamTimer);
        streamTimer = setTimeout(() => {
          if (disposed) return;
          setError('No perception data received. The backend may be starting up or unavailable.');
          socket?.close();
        }, STREAM_TIMEOUT_MS);
      };
      try {
        socket = new WebSocket(url.href);
      } catch {
        setReconnecting(false);
        setError('Unable to open the backend connection. Check the backend URL and availability.');
        return;
      }
      armTimeout();
      socket.onmessage = event => {
        if (disposed) return;
        try {
          const parsed: WSMessage = JSON.parse(event.data);
          if (!parsed.driver || !parsed.road || !parsed.metrics || !parsed.risk_data) return;
          setData(parsed);
          setConnected(true);
          setReconnecting(false);
          setError(null);
          retries = 0;
          armTimeout();
        } catch { /* Wait for a valid perception message. */ }
      };
      socket.onclose = () => {
        clearTimeout(streamTimer);
        if (disposed) return;
        setConnected(false);
        setError('Perception stream disconnected. Check that the Python backend is running.');
        if (retries < MAX_RETRIES) {
          setReconnecting(true);
          retryTimer = setTimeout(connect, Math.min(1000 * 2 ** retries++, 16000));
        } else {
          setReconnecting(false);
        }
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => {
      disposed = true;
      clearTimeout(retryTimer);
      clearTimeout(streamTimer);
      if (socket) {
        socket.onclose = null;
        socket.onerror = null;
        socket.onmessage = null;
        socket.close();
      }
    };
  }, [attempt]);

  return { data, connected, reconnecting, error, retry };
}
