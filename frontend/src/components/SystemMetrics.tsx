import { useEffect, useState } from 'react';
import { SystemMetricsData } from '../types';

interface Props {
  metrics: SystemMetricsData;
  connected: boolean;
  reconnecting: boolean;
}

export function SystemMetrics({ metrics, connected, reconnecting }: Props) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const StatusDot = ({ active, blink = false }: { active: boolean; blink?: boolean }) => (
    <div className={`w-2 h-2 rounded-full ${active ? 'bg-green-500' : 'bg-red-500'} ${blink ? 'animate-pulse' : ''}`} />
  );

  return (
    <div className="flex items-center justify-between text-xs font-mono text-slate-400 border-t border-[#334155] p-2 mt-4 bg-[#0a0f1e]">
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <span>WS:</span>
          <StatusDot active={connected} blink={reconnecting} />
          <span className={connected ? 'text-green-500' : reconnecting ? 'text-yellow-500' : 'text-red-500'}>
            {connected ? 'CONNECTED' : reconnecting ? 'RECONNECTING...' : 'OFFLINE'}
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          <span>DRIVER CAM:</span>
          <StatusDot active={metrics.driver_connected} />
        </div>
        
        <div className="flex items-center space-x-2">
          <span>ROAD CAM:</span>
          <StatusDot active={metrics.road_connected} />
        </div>
      </div>
      
      <div className="flex items-center space-x-6">
        <div>FPS: <span className="text-white">{metrics.fps.toFixed(1)}</span></div>
        <div>LATENCY: <span className="text-white">{metrics.latency_ms.toFixed(0)}ms</span></div>
        <div className="text-white">{time.toLocaleTimeString()}</div>
      </div>
    </div>
  );
}
