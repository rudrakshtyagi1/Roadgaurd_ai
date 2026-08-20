import { useState } from 'react';
import { useMockData } from './hooks/useMockData';
import { useWebSocket } from './hooks/useWebSocket';
import { DriverMonitor } from './components/DriverMonitor';
import { RoadMonitor } from './components/RoadMonitor';
import { RiskGauge } from './components/RiskGauge';
import { FatigueMeter } from './components/FatigueMeter';
import { HazardPanel } from './components/HazardPanel';
import { AlertBanner } from './components/AlertBanner';
import { EventTimeline } from './components/EventTimeline';
import { SystemMetrics } from './components/SystemMetrics';

function App() {
  const [useMock, setUseMock] = useState(true);
  
  const mock = useMockData();
  const ws = useWebSocket();
  
  const current = useMock ? mock : ws;
  
  if (!current.data) {
    return (
      <div className="min-h-screen bg-[#0a0f1e] text-white flex items-center justify-center font-mono">
        <div className="text-center">
          <div className="text-4xl mb-4 animate-pulse">🚗</div>
          <div className="text-xl text-slate-400">INITIALIZING SYSTEM...</div>
        </div>
      </div>
    );
  }

  const { driver, road, risk_data, metrics, events, temporal } = current.data;

  return (
    <div className="min-h-screen bg-[#0a0f1e] text-white p-4 flex flex-col font-mono max-w-7xl mx-auto">
      
      {/* Header */}
      <div className="flex justify-between items-center mb-4 bg-[#1e293b] p-4 rounded-xl border border-[#334155]">
        <div className="flex items-center space-x-4">
          <span className="text-2xl">🚗</span>
          <h1 className="text-2xl font-bold tracking-widest text-slate-200">ROADGUARD AI</h1>
          <span className={`px-3 py-1 rounded text-xs font-bold ${risk_data.status === 'SAFE' ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}`}>
            SYSTEM {risk_data.status}
          </span>
        </div>
        
        <button 
          onClick={() => setUseMock(!useMock)}
          className="flex items-center space-x-2 bg-[#0f172a] px-4 py-2 rounded-lg border border-slate-600 hover:bg-slate-800 transition-colors"
        >
          <div className={`w-2 h-2 rounded-full ${useMock ? 'bg-yellow-500' : 'bg-green-500'}`} />
          <span className="text-sm font-bold">{useMock ? 'MOCK DATA' : 'LIVE'}</span>
        </button>
      </div>
      
      <AlertBanner riskData={risk_data} />
      
      {/* Main Content Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow mb-4">
        
        {/* Left Column */}
        <div className="flex flex-col space-y-4">
          <DriverMonitor driver={driver} temporal={temporal} />
          <FatigueMeter driver={driver} temporal={temporal} />
        </div>
        
        {/* Right Column */}
        <div className="flex flex-col space-y-4">
          <RoadMonitor road={road} />
          <HazardPanel road={road} />
        </div>
        
      </div>
      
      {/* Central Gauge row */}
      <div className="bg-[#1e293b] border border-[#334155] rounded-xl p-4 flex flex-col items-center justify-center mb-4">
        <h2 className="text-slate-400 font-bold mb-2 tracking-wider">OVERALL SYSTEM RISK</h2>
        <RiskGauge risk={risk_data.risk} status={risk_data.status} />
        
        {/* Risk Components Bar */}
        <div className="w-full max-w-2xl mt-4 grid grid-cols-4 gap-2 text-center text-xs">
          <div className="bg-[#0f172a] p-2 rounded">
            <div className="text-slate-400 mb-1">Fatigue</div>
            <div className="text-orange-400 font-bold">{risk_data.components.fatigue_score.toFixed(0)}</div>
          </div>
          <div className="bg-[#0f172a] p-2 rounded">
            <div className="text-slate-400 mb-1">Attention</div>
            <div className="text-yellow-400 font-bold">{risk_data.components.attention_score.toFixed(0)}</div>
          </div>
          <div className="bg-[#0f172a] p-2 rounded">
            <div className="text-slate-400 mb-1">Hazard</div>
            <div className="text-red-400 font-bold">{risk_data.components.hazard_score.toFixed(0)}</div>
          </div>
          <div className="bg-[#0f172a] p-2 rounded">
            <div className="text-slate-400 mb-1">Proximity</div>
            <div className="text-blue-400 font-bold">{risk_data.components.proximity_score.toFixed(0)}</div>
          </div>
        </div>
      </div>
      
      <EventTimeline events={events} />
      
      <SystemMetrics metrics={metrics} connected={current.connected} reconnecting={current.reconnecting} />
      
    </div>
  );
}

export default App;
