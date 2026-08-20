import { DriverState, TemporalStats } from '../types';

interface Props {
  driver: DriverState;
  temporal: TemporalStats;
}

export function DriverMonitor({ driver, temporal: _temporal }: Props) {
  const getFatigueColor = (val: number) => {
    if (val >= 0.75) return 'bg-red-500';
    if (val >= 0.6) return 'bg-orange-500';
    if (val >= 0.4) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getAttentionColor = (level: string) => {
    switch (level) {
      case 'HIGH': return 'text-green-400 border-green-400';
      case 'MEDIUM': return 'text-yellow-400 border-yellow-400';
      case 'LOW': return 'text-orange-400 border-orange-400';
      case 'DISTRACTED': return 'text-red-400 border-red-400';
      default: return 'text-gray-400 border-gray-400';
    }
  };

  const eyeIcon = driver.eye_state === 'CLOSED' ? '💤' : driver.eye_state === 'CLOSING' ? '😴' : '👁️';

  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-4 flex flex-col h-full">
      <h2 className="text-slate-400 font-bold mb-4 tracking-wider">🧑 DRIVER MONITOR</h2>
      
      <div className="grid grid-cols-2 gap-4 flex-grow">
        <div className="flex flex-col justify-center items-center p-4 bg-[#0f172a] rounded-lg">
          <div className="text-5xl mb-2">{eyeIcon}</div>
          <div className="text-xl font-bold">{driver.eye_state}</div>
          <div className="text-sm text-slate-400 mt-2">EAR: {driver.ear.toFixed(3)}</div>
        </div>
        
        <div className="flex flex-col justify-between space-y-4">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>Fatigue</span>
              <span>{(driver.fatigue * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div className={`h-2 rounded-full transition-all duration-300 ${getFatigueColor(driver.fatigue)}`} style={{ width: `${Math.min(100, driver.fatigue * 100)}%` }}></div>
            </div>
          </div>
          
          <div>
            <div className="text-sm text-slate-400 mb-1">Attention Level</div>
            <span className={`inline-block px-3 py-1 border rounded-full text-xs font-bold ${getAttentionColor(driver.attention)}`}>
              {driver.attention}
            </span>
          </div>
          
          <div>
            <div className="text-sm text-slate-400 mb-2">Head Pose</div>
            <div className="space-y-1">
              <div className="flex items-center text-xs">
                <span className="w-8">Yaw</span>
                <div className="flex-1 bg-slate-700 h-1 mx-2 relative"><div className="absolute top-0 h-1 bg-blue-400" style={{ left: '50%', width: `${Math.abs(driver.head_pose.yaw)}%`, transform: driver.head_pose.yaw < 0 ? 'translateX(-100%)' : '' }}></div></div>
              </div>
              <div className="flex items-center text-xs">
                <span className="w-8">Pitch</span>
                <div className="flex-1 bg-slate-700 h-1 mx-2 relative"><div className="absolute top-0 h-1 bg-green-400" style={{ left: '50%', width: `${Math.abs(driver.head_pose.pitch)}%`, transform: driver.head_pose.pitch < 0 ? 'translateX(-100%)' : '' }}></div></div>
              </div>
              <div className="flex items-center text-xs">
                <span className="w-8">Roll</span>
                <div className="flex-1 bg-slate-700 h-1 mx-2 relative"><div className="absolute top-0 h-1 bg-purple-400" style={{ left: '50%', width: `${Math.abs(driver.head_pose.roll)}%`, transform: driver.head_pose.roll < 0 ? 'translateX(-100%)' : '' }}></div></div>
              </div>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}
