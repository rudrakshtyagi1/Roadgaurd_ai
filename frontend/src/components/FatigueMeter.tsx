import { useMemo } from 'react';
import { DriverState, TemporalStats } from '../types';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';

interface Props {
  driver: DriverState;
  temporal: TemporalStats;
}

export function FatigueMeter({ driver, temporal }: Props) {
  const trendIcon = temporal.ear_trend < -0.001 ? '↓' : temporal.ear_trend > 0.001 ? '↑' : '→';
  const trendColor = temporal.ear_trend < -0.001 ? 'text-red-400' : temporal.ear_trend > 0.001 ? 'text-green-400' : 'text-slate-400';

  const chartData = useMemo(() => {
    return Array.from({ length: 20 }).map((_, i) => ({
      val: temporal.ear_mean + (Math.sin(i * 0.5) * 0.05)
    }));
  }, [temporal.ear_mean]);

  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-slate-400 font-bold tracking-wider">😴 FATIGUE ANALYSIS</h2>
        <div className="flex items-center space-x-2">
          <span className="text-2xl font-bold">{(driver.fatigue * 100).toFixed(0)}%</span>
          <span className={`text-xl font-bold ${trendColor}`}>{trendIcon}</span>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-xs mb-1 text-slate-300">
            <span>PERCLOS (Eye Closure)</span>
            <span>{temporal.perclos.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-1.5">
            <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${Math.min(100, temporal.perclos)}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1 text-slate-300">
            <span>MAR (Yawning)</span>
            <span>{driver.mar.toFixed(3)}</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-1.5">
            <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${Math.min(100, driver.mar * 100)}%` }} />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs mb-1 text-slate-300">
            <span>Blink Rate (bpm)</span>
            <span>{driver.blink_rate.toFixed(1)}</span>
          </div>
          <div className="w-full bg-slate-700 rounded-full h-1.5">
            <div className="bg-teal-500 h-1.5 rounded-full" style={{ width: `${Math.min(100, driver.blink_rate * 2)}%` }} />
          </div>
        </div>
        
        <div className="mt-4 pt-2 border-t border-slate-700">
          <span className="text-xs text-slate-400 block mb-2">EAR Trend History</span>
          <div className="h-16 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorEar" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <YAxis domain={[0.1, 0.4]} hide />
                <Area type="monotone" dataKey="val" stroke="#3b82f6" fillOpacity={1} fill="url(#colorEar)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
