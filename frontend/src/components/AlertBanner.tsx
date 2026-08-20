import { RiskData } from '../types';

interface Props {
  riskData: RiskData;
}

export function AlertBanner({ riskData }: Props) {
  if (riskData.status === 'SAFE') {
    return <div className="h-0 overflow-hidden transition-all duration-500 ease-in-out" />;
  }

  const getStyle = () => {
    switch (riskData.status) {
      case 'CAUTION': return 'bg-yellow-500 text-black';
      case 'HIGH': return 'bg-orange-500 text-white animate-pulse';
      case 'CRITICAL': return 'bg-red-600 text-white font-bold animate-pulse-glow';
      default: return '';
    }
  };

  const icon = riskData.status === 'CRITICAL' ? '🚨' : '⚠️';

  return (
    <div className={`transition-all duration-500 ease-in-out mb-4 rounded-lg p-4 flex items-center justify-center space-x-4 ${getStyle()}`}>
      <span className="text-3xl">{icon}</span>
      <span className="text-xl tracking-wide">{riskData.alert}</span>
      <span className="text-3xl">{icon}</span>
    </div>
  );
}
