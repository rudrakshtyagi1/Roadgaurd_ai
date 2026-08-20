import { useEffect, useRef } from 'react';
import { RoadGuardEvent, RiskStatus } from '../types';

interface Props {
  events: RoadGuardEvent[];
}

export function EventTimeline({ events }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  const getColor = (status: RiskStatus) => {
    switch (status) {
      case 'SAFE': return 'bg-green-500';
      case 'CAUTION': return 'bg-yellow-500';
      case 'HIGH': return 'bg-orange-500';
      case 'CRITICAL': return 'bg-red-500';
      default: return 'bg-slate-500';
    }
  };

  return (
    <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-4 mt-4">
      <h2 className="text-slate-400 font-bold mb-4 tracking-wider">📋 EVENT LOG</h2>
      
      <div 
        ref={scrollRef}
        className="max-h-[180px] overflow-y-auto pr-2 space-y-2 scrollbar-thin scrollbar-thumb-slate-600"
      >
        {events.length === 0 ? (
          <div className="text-slate-500 text-center py-4 text-sm">No events logged</div>
        ) : (
          events.slice(-15).map(evt => (
            <div key={evt.id} className="flex items-center space-x-3 text-sm p-2 hover:bg-[#0f172a] rounded transition-colors">
              <span className="text-slate-400 font-mono text-xs">{new Date(evt.timestamp).toLocaleTimeString()}</span>
              <div className={`w-2 h-2 rounded-full ${getColor(evt.risk_level)}`} />
              <span className="text-slate-300 font-mono text-xs w-24 truncate">{evt.type}</span>
              <span className="text-white flex-grow">{evt.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
