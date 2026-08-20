import { useState, useEffect } from 'react';
import { WSMessage, RiskStatus } from '../types';

export function useMockData() {
  const [data, setData] = useState<WSMessage | null>(null);

  useEffect(() => {
    let t = 0;
    
    const interval = setInterval(() => {
      t += 0.5;
      
      const ear = 0.25 + 0.1 * Math.sin(t * 0.5);
      const fatigueBase = 0.4 + 0.45 * (t / 100); 
      const fatigue = Math.min(0.85, Math.max(0.4, fatigueBase + 0.1 * Math.sin(t * 0.2)));
      
      const vehicleCount = Math.max(0, Math.floor(3 + 3 * Math.sin(t * 0.3)));
      const isPothole = t % 8 < 1; 
      
      const hazardScore = Math.min(100, vehicleCount * 10 + (isPothole ? 40 : 0));
      
      const riskVal = Math.min(100, Math.max(0, Math.round(fatigue * 50 + hazardScore * 0.5)));
      let status: RiskStatus = 'SAFE';
      let alert = '✅ All systems normal — drive safe';
      if (riskVal > 75) { status = 'CRITICAL'; alert = '🚨 CRITICAL — REDUCE SPEED IMMEDIATELY'; }
      else if (riskVal > 55) { status = 'HIGH'; alert = '⚠️ HIGH RISK — Reduce speed and take a break'; }
      else if (riskVal > 30) { status = 'CAUTION'; alert = '⚠️ CAUTION — Stay alert, conditions changing'; }
      
      const newMessage: WSMessage = {
        timestamp: new Date().toISOString(),
        driver: {
          ear: ear,
          mar: 0.1 + 0.05 * Math.sin(t),
          head_pose: { yaw: 5 * Math.sin(t), pitch: 2 * Math.cos(t), roll: 1 * Math.sin(t*0.5) },
          fatigue: fatigue,
          attention: fatigue > 0.7 ? 'LOW' : fatigue > 0.5 ? 'MEDIUM' : 'HIGH',
          eye_state: ear < 0.2 ? 'CLOSED' : ear < 0.25 ? 'CLOSING' : 'OPEN',
          blink_rate: 15 + 5 * Math.sin(t * 0.1),
          face_detected: true
        },
        road: {
          potholes: isPothole ? [{ confidence: 0.85, severity: 'HIGH', distance_m: 15 }] : [],
          vehicles: vehicleCount,
          hazard_score: hazardScore,
          detections: [],
          nearest_distance_m: 20 + 10 * Math.sin(t),
          pothole_detected: isPothole,
          pothole_confidence: isPothole ? 0.85 : 0
        },
        risk_data: {
          risk: riskVal,
          status: status,
          alert: alert,
          components: {
            fatigue_score: fatigue * 100,
            attention_score: (1 - fatigue) * 100,
            hazard_score: hazardScore,
            proximity_score: 50
          }
        },
        metrics: {
          fps: 30 + Math.random() * 5,
          latency_ms: 45 + Math.random() * 10,
          driver_connected: true,
          road_connected: true
        },
        events: [],
        temporal: {
          ear_mean: ear + 0.02,
          ear_trend: -0.002,
          perclos: fatigue * 20,
          mar_mean: 0.1,
          blink_rate: 15,
          vehicle_count_mean: 3
        }
      };
      
      setData(newMessage);
    }, 500);
    
    return () => clearInterval(interval);
  }, []);

  return { data, connected: true, reconnecting: false };
}
