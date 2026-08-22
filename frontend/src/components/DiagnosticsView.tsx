import { useEffect, useState } from 'react';
import { WSMessage, RoadGuardEvent } from '../types';

interface Props {
  data: WSMessage;
  connected: boolean;
  reconnecting: boolean;
}

export function DiagnosticsView({ data, connected, reconnecting }: Props) {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const StatusDot = ({ active, blink = false }: { active: boolean; blink?: boolean }) => (
    <div className={`w-2 h-2 rounded-full ${active ? 'bg-green-500' : 'bg-red-500'} ${blink ? 'animate-pulse' : ''}`} />
  );

  const getEventStyle = (level: string) => {
    switch (level) {
      case 'CRITICAL': return 'text-red-500 font-bold';
      case 'HIGH': return 'text-orange-500 font-bold';
      case 'CAUTION': return 'text-amber-500';
      default: return 'text-slate-400';
    }
  };

  const { driver, road, metrics, events, temporal } = data;

  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-6 font-mono text-xs text-slate-300 space-y-6">
      <div className="flex justify-between items-center border-b border-[#334155] pb-3">
        <h3 className="text-slate-400 font-bold tracking-widest uppercase">🛠️ DIAGNOSTICS & TELEMETRY</h3>
        <span className="text-[10px] text-slate-500">{time.toLocaleTimeString()}</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Column 1: Driver Diagnostics */}
        <div className="space-y-4">
          <h4 className="text-slate-400 font-bold border-b border-slate-700 pb-1">DRIVER TELEMETRY</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Eye Aspect Ratio (EAR):</span>
              <span className="text-white font-bold">{driver.ear.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Mouth Aspect Ratio (MAR):</span>
              <span className="text-white">{driver.mar.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>PERCLOS (Eye Closure Rate):</span>
              <span className="text-white">{(temporal.perclos * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Blink Rate:</span>
              <span className="text-white">{driver.blink_rate} bpm</span>
            </div>
            
            <div className="pt-2">
              <span className="text-slate-500 block mb-1">Head Pose Vector (Euler):</span>
              <div className="grid grid-cols-3 gap-1 bg-[#0f172a] p-2 rounded text-center text-[10px]">
                <div>Yaw: <span className="text-blue-400 font-bold">{driver.head_pose.yaw.toFixed(1)}°</span></div>
                <div>Pitch: <span className="text-green-400 font-bold">{driver.head_pose.pitch.toFixed(1)}°</span></div>
                <div>Roll: <span className="text-purple-400 font-bold">{driver.head_pose.roll.toFixed(1)}°</span></div>
              </div>
            </div>
          </div>
        </div>

        {/* Column 2: Road Diagnostics */}
        <div className="space-y-4">
          <h4 className="text-slate-400 font-bold border-b border-slate-700 pb-1">ROAD TELEMETRY</h4>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Active Detections:</span>
              <span className="text-white font-bold">{road.potholes.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Raw Confidence:</span>
              <span className="text-white">{road.pothole_confidence.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span>Relative Proximity:</span>
              <span className="text-white">{road.relative_proximity || 'FAR'}</span>
            </div>
            <div className="flex justify-between">
              <span>Persistence Counter:</span>
              <span className="text-white">{road.pothole_detected ? "CONFIRMED" : "STABILIZING"}</span>
            </div>
            <div className="pt-2">
              <span className="text-slate-500 block mb-1">Raw Confidences:</span>
              <div className="bg-[#0f172a] p-2 rounded text-slate-400 font-mono text-[10px] min-h-[40px] break-all">
                {road.potholes.length > 0 
                  ? road.potholes.map(p => p.confidence.toFixed(2)).join(', ')
                  : "No potholes in view"}
              </div>
            </div>
          </div>
        </div>

        {/* Column 3: System Context */}
        <div className="space-y-4">
          <h4 className="text-slate-400 font-bold border-b border-slate-700 pb-1">SYSTEM CONTEXT</h4>
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span>WebSocket Stream:</span>
              <div className="flex items-center space-x-2">
                <StatusDot active={connected} blink={reconnecting} />
                <span className={connected ? 'text-green-500' : reconnecting ? 'text-yellow-500 animate-pulse' : 'text-red-500'}>
                  {connected ? 'CONNECTED' : reconnecting ? 'RECONNECTING' : 'OFFLINE'}
                </span>
              </div>
            </div>
            <div className="flex justify-between">
              <span>Inference Device:</span>
              <span className="text-white">Apple MPS (GPU)</span>
            </div>
            <div className="flex justify-between">
              <span>Inference Latency:</span>
              <span className="text-white font-bold">{metrics.latency_ms.toFixed(1)} ms</span>
            </div>
            <div className="flex justify-between">
              <span>Loop Execution Speed:</span>
              <span className="text-white font-bold">{metrics.fps.toFixed(1)} FPS</span>
            </div>
            <div className="flex justify-between">
              <span>Network Protocol:</span>
              <span className="text-white">JSON / WS v1.0</span>
            </div>
          </div>
        </div>

      </div>

      {/* Compact Event Timeline Log */}
      <div className="border-t border-[#334155] pt-4">
        <h4 className="text-slate-400 font-bold mb-2">📋 SYSTEM LOG TIMELINE</h4>
        <div className="bg-[#0f172a] rounded-lg p-3 max-h-[120px] overflow-y-auto font-mono text-[10px] space-y-1 divide-y divide-slate-800/40">
          {events.length === 0 ? (
            <div className="text-slate-500 italic text-center py-2">No alerts logged in the current session.</div>
          ) : (
            events.map((e: RoadGuardEvent) => (
              <div key={e.id} className="flex justify-between py-1 items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <span className="text-slate-500">{new Date(e.timestamp).toLocaleTimeString()}</span>
                  <span className="text-slate-400">•</span>
                  <span className={getEventStyle(e.risk_level)}>[{e.type}]</span>
                  <span className="text-slate-300">{e.message}</span>
                </div>
                <span className={`px-2 py-0.5 rounded text-[8px] font-bold ${getEventStyle(e.risk_level)} bg-slate-900`}>
                  {e.risk_level}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
