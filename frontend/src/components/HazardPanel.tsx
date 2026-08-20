
import { RoadState } from '../types';

interface Props {
  road: RoadState;
}

export function HazardPanel({ road }: Props) {
  const getSeverityColor = (sev: string) => {
    switch(sev) {
      case 'HIGH': return 'bg-red-500 text-white';
      case 'MEDIUM': return 'bg-orange-500 text-white';
      case 'LOW': return 'bg-yellow-500 text-black';
      default: return 'bg-slate-500 text-white';
    }
  };

  const isWarning = road.hazard_score > 60;

  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-slate-400 font-bold tracking-wider">⚠️ HAZARD ANALYSIS</h2>
        {isWarning && <span className="text-2xl animate-pulse">⚠️</span>}
      </div>

      <div className="space-y-4">
        <div className="bg-[#0f172a] p-3 rounded-lg border border-slate-700">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-slate-300">Total Hazard Score</span>
            <span className={`font-bold ${isWarning ? 'text-orange-500' : 'text-green-500'}`}>{road.hazard_score.toFixed(0)}</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-2">
            <div className={`h-2 rounded-full ${isWarning ? 'bg-orange-500' : 'bg-green-500'}`} style={{ width: `${Math.min(100, road.hazard_score)}%` }} />
          </div>
        </div>

        <div className="bg-[#0f172a] p-3 rounded-lg border border-slate-700">
          <span className="text-xs text-slate-400 block mb-2">Detected Hazards</span>
          {road.potholes.length === 0 ? (
            <div className="text-sm text-green-500 font-bold">✓ Clear Road Ahead</div>
          ) : (
            <div className="space-y-2">
              {road.potholes.map((p, i) => (
                <div key={i} className="flex justify-between items-center text-sm">
                  <div className="flex items-center space-x-2">
                    <span>🕳️</span>
                    <span className={`text-xs px-2 py-0.5 rounded ${getSeverityColor(p.severity)}`}>
                      Pothole ({p.severity})
                    </span>
                  </div>
                  <span className="text-slate-300">{p.distance_m.toFixed(1)}m</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-[#0f172a] p-3 rounded-lg border border-slate-700">
          <div className="flex justify-between items-center">
            <span className="text-sm text-slate-300">Nearest Object</span>
            <span className={`font-bold ${road.nearest_distance_m < 20 ? 'text-red-400' : 'text-slate-300'}`}>
              {road.nearest_distance_m.toFixed(1)}m
            </span>
          </div>
          {road.nearest_distance_m < 20 && (
            <div className="mt-2 text-xs text-red-400 text-right animate-pulse">TOO CLOSE</div>
          )}
        </div>
      </div>
    </div>
  );
}
