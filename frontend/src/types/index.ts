export type EyeState = 'OPEN' | 'CLOSING' | 'CLOSED';
export type AttentionLevel = 'HIGH' | 'MEDIUM' | 'LOW' | 'DISTRACTED';
export type RiskStatus = 'SAFE' | 'CAUTION' | 'HIGH' | 'CRITICAL';
export type Severity = 'LOW' | 'MEDIUM' | 'HIGH';
export type AlertType = 'FATIGUE_WARNING' | 'ATTENTION_WARNING' | 'POTHOLE_DETECTED' | 'HIGH_RISK' | 'CRITICAL_RISK';

export interface HeadPose {
  yaw: number;
  pitch: number;
  roll: number;
}

export interface DriverState {
  ear: number;
  mar: number;
  head_pose: HeadPose;
  fatigue: number;
  attention: AttentionLevel;
  eye_state: EyeState;
  blink_rate: number;
  face_detected: boolean;
}

export interface Pothole {
  confidence: number;
  severity: Severity;
  distance_m: number;
}

export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  distance_m: number;
}

export interface RoadState {
  potholes: Pothole[];
  vehicles: number;
  hazard_score: number;
  detections: Detection[];
  nearest_distance_m: number;
  pothole_detected: boolean;
  pothole_confidence: number;
}

export interface RiskComponents {
  fatigue_score: number;
  attention_score: number;
  hazard_score: number;
  proximity_score: number;
}

export interface RiskData {
  risk: number;
  status: RiskStatus;
  alert: string;
  components: RiskComponents;
}

export interface SystemMetricsData {
  fps: number;
  latency_ms: number;
  driver_connected: boolean;
  road_connected: boolean;
}

export interface RoadGuardEvent {
  id: string;
  timestamp: string;
  type: AlertType;
  message: string;
  risk_level: RiskStatus;
}

export interface TemporalStats {
  ear_mean: number;
  ear_trend: number;
  perclos: number;
  mar_mean: number;
  blink_rate: number;
  vehicle_count_mean: number;
}

export interface WSMessage {
  timestamp: string;
  driver: DriverState;
  road: RoadState;
  risk_data: RiskData;
  metrics: SystemMetricsData;
  events: RoadGuardEvent[];
  temporal: TemporalStats;
}
