import { useEffect, useState } from 'react';
import { RiskStatus } from '../types';

interface Props {
  risk: number;
  status: RiskStatus;
}

export function RiskGauge({ risk, status }: Props) {
  const [animatedRisk, setAnimatedRisk] = useState(risk);

  useEffect(() => {
    setAnimatedRisk(risk);
  }, [risk]);

  const getColor = (status: RiskStatus) => {
    switch (status) {
      case 'SAFE': return '#22c55e';
      case 'CAUTION': return '#eab308';
      case 'HIGH': return '#f97316';
      case 'CRITICAL': return '#ef4444';
      default: return '#334155';
    }
  };
  
  const getGlow = (status: RiskStatus) => {
    switch (status) {
      case 'SAFE': return 'drop-shadow(0 0 10px rgba(34,197,94,0.5))';
      case 'CAUTION': return 'drop-shadow(0 0 10px rgba(234,179,8,0.5))';
      case 'HIGH': return 'drop-shadow(0 0 10px rgba(249,115,22,0.5))';
      case 'CRITICAL': return 'drop-shadow(0 0 15px rgba(239,68,68,0.8))';
      default: return 'none';
    }
  };

  const radius = 80;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degrees
  const strokeDashoffset = arcLength - (animatedRisk / 100) * arcLength;

  return (
    <div className="flex flex-col items-center justify-center p-6 relative">
      <div className="relative w-[200px] h-[200px]" style={{ filter: getGlow(status) }}>
        <svg viewBox="0 0 200 200" className="w-full h-full transform rotate-[135deg]">
          {/* Background arc */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke="#1e293b"
            strokeWidth="16"
            strokeDasharray={arcLength + ' ' + circumference}
            strokeLinecap="round"
          />
          {/* Foreground arc */}
          <circle
            cx="100"
            cy="100"
            r={radius}
            fill="none"
            stroke={getColor(status)}
            strokeWidth="16"
            strokeDasharray={arcLength + ' ' + circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-500 ease-out"
          />
        </svg>
        
        {/* Center Text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none mt-4">
          <span className="text-5xl font-bold text-white tracking-tighter" style={{ color: getColor(status) }}>
            {Math.round(animatedRisk)}
          </span>
          <span className="text-sm font-bold mt-1 tracking-widest uppercase" style={{ color: getColor(status) }}>
            {status}
          </span>
        </div>
      </div>
    </div>
  );
}
