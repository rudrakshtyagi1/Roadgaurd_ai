
import { RoadState } from '../types';

interface Props {
  road: RoadState;
}

export function RoadMonitor({ road }: Props) {
  const getHazardColor = (score: number) => {
    if (score >= 75) return 'text-red-500';
    if (score >= 50) return 'text-orange-500';
    if (score >= 25) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity) {
      case 'HIGH': return 'bg-red-500 text-white';
      case 'MEDIUM': return 'bg-orange-500 text-white';
      case 'LOW': return 'bg-yellow-500 text-black';
      default: return 'bg-gray-500 text-white';
    }
  };

  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-4 flex flex-col h-full">
      <h2 className="text-slate-400 font-bold mb-4 tracking-wider">🛣️ ROAD MONITOR</h2>
      
      <div className="grid grid-cols-2 gap-4 flex-grow">
        <div className="flex flex-col space-y-4">
          <div className="bg-[#0f172a] p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-slate-400">Pothole Status</span>
            <div className="flex items-center space-x-2">
              {road.pothole_detected && <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse-glow" />}
              <span className={`font-bold ${road.pothole_detected ? 'text-red-500' : 'text-green-500'}`}>
                {road.pothole_detected ? 'DETECTED' : 'CLEAR'}
              </span>
            </div>
          </div>
          
          <div className="bg-[#0f172a] p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-slate-400">Vehicles</span>
            <div className="flex items-center space-x-2">
              <span>🚗</span>
              <span className="font-bold text-xl">{road.vehicles}</span>
            </div>
          </div>
          
          <div className="bg-[#0f172a] p-3 rounded-lg flex items-center justify-between">
            <span className="text-sm text-slate-400">Nearest</span>
            <div className="flex items-center space-x-2">
              <span>📏</span>
              <span className="font-bold">{road.nearest_distance_m.toFixed(1)}m</span>
            </div>
          </div>
        </div>
        
        <div className="flex flex-col">
          <div className="bg-[#0f172a] p-4 rounded-lg flex flex-col items-center justify-center flex-grow mb-4">
            <span className="text-sm text-slate-400 mb-1">Hazard Score</span>
            <span className={`text-4xl font-bold ${getHazardColor(road.hazard_score)}`}>
              {road.hazard_score.toFixed(0)}
            </span>
          </div>
          
          {road.potholes.length > 0 && (
            <div className="bg-[#0f172a] p-2 rounded-lg">
              <span className="text-xs text-slate-400 block mb-2">Active Potholes</span>
              {road.potholes.map((p, i) => (
                <div key={i} className="flex justify-between items-center text-xs mb-1">
                  <span className={`px-2 py-0.5 rounded ${getSeverityBadge(p.severity)}`}>{p.severity}</span>
                  <span>{p.distance_m.toFixed(1)}m</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
