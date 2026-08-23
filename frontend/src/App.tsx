import { useState, useRef, useEffect } from 'react';
import { useMockData } from './hooks/useMockData';
import { useWebSocket } from './hooks/useWebSocket';
import { DiagnosticsView } from './components/DiagnosticsView';

interface LiveDriverInference {
  drowsy_probability: number;
  cnn_drowsy_probability?: number;
  cnn_predicted_class?: string;
  cnn_predicted_class_probability?: number;
  smoothed_fatigue: number;
  state: 'NORMAL' | 'FATIGUE_RISK' | 'DROWSY' | 'CRITICAL' | 'ALERT' | 'LOW_VIGILANCE';
  ear: number;
  mar: number;
  eye_state: 'OPEN' | 'CLOSING' | 'CLOSED';
  blink_count: number;
  avg_blink_duration_ms: number;
  face_detected: boolean;
  reason_codes?: string[];
  fatigue_risk_score?: number;
  signal_strength?: number;
  tracking_state?: string;
  wakefulness_support?: number;
  signal_disagreement?: boolean;
  active_signals?: {
    eyes_closed?: boolean;
    active_yawn?: boolean;
    head_drop?: boolean;
  };
  recent_signals?: {
    long_blinks_30s?: number;
    yawns_60s?: number;
    effective_long_blink_weight?: number;
    effective_yawn_weight?: number;
  };
  driver_state_v2?: any;
  latency_ms: number;
  is_mock?: boolean;
}

function App() {
  const [useMock, setUseMock] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  
  // Live webcam testing state
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [liveDriverState, setLiveDriverState] = useState<LiveDriverInference | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  const mock = useMockData();
  const ws = useWebSocket();
  const current = useMock ? mock : ws;

  // ── Start / Stop Driver Camera ─────────────────────────────────────────────
  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsCameraActive(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setCameraError(err.message || 'Camera permission denied or camera unavailable');
      setIsCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setLiveDriverState(null);
  };

  // ── Live Frame Capture & Inference Dispatch (~10 FPS) ──────────────────────
  useEffect(() => {
    if (!isCameraActive) return;

    let isProcessing = false;
    const interval = setInterval(() => {
      if (isProcessing || !videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (video.readyState < 2) return; // Wait until current data is available

      isProcessing = true;
      try {
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          isProcessing = false;
          return;
        }

        ctx.drawImage(video, 0, 0, 320, 240);
        canvas.toBlob(
          async (blob) => {
            if (!blob) {
              isProcessing = false;
              return;
            }

            const formData = new FormData();
            formData.append('frame', blob, 'driver_frame.jpg');

            try {
              const res = await fetch('http://localhost:8000/api/video/driver', {
                method: 'POST',
                body: formData,
              });

              if (res.ok) {
                const result = await res.json();
                setLiveDriverState(result);
              }
            } catch (err) {
              console.warn('Driver frame inference request failed:', err);
            } finally {
              isProcessing = false;
            }
          },
          'image/jpeg',
          0.85
        );
      } catch (e) {
        isProcessing = false;
      }
    }, 100);

    return () => {
      clearInterval(interval);
    };
  }, [isCameraActive]);

  if (!current.data) {
    return (
      <div className="min-h-screen bg-[#07090e] text-slate-300 flex items-center justify-center font-mono">
        <div className="text-center space-y-3">
          <div className="w-3 h-3 rounded-full bg-emerald-500 animate-ping mx-auto" />
          <div className="text-xs uppercase tracking-widest text-slate-400">Connecting to Perception Stream...</div>
        </div>
      </div>
    );
  }

  const { road, metrics } = current.data;

  // ── Perception & Risk Derivations ──────────────────────────────────────────
  const isPothole = road.pothole_detected;
  const proximity = road.relative_proximity || 'FAR';
  const potholeCount = road.potholes ? road.potholes.length : 0;
  const maxConfPct = road.pothole_confidence ? (road.pothole_confidence * 100).toFixed(0) : '0';

  // Effective Driver State (Live webcam takes priority if active)
  const isDriverLive = isCameraActive && liveDriverState !== null;
  const driverState = isDriverLive
    ? liveDriverState.state
    : current.data.driver.is_mock
    ? 'STANDBY'
    : current.data.driver.eye_state === 'CLOSED'
    ? 'DROWSY'
    : 'ALERT';

  const driverProb = isDriverLive ? liveDriverState.drowsy_probability : 0.0;
  const driverSmoothed = isDriverLive ? liveDriverState.smoothed_fatigue : 0.0;

  // 1. Road Status
  let roadTitle = 'CLEAR';
  let roadSubtext = 'No road hazards detected';
  let roadColor = 'text-emerald-400';

  if (isPothole) {
    if (proximity === 'NEAR') {
      roadTitle = 'HIGH HAZARD';
      roadColor = 'text-red-400';
    } else if (proximity === 'MEDIUM') {
      roadTitle = 'CONFIRMED POTHOLE';
      roadColor = 'text-amber-400';
    } else {
      roadTitle = 'POTENTIAL HAZARD';
      roadColor = 'text-amber-300';
    }
    roadSubtext = `${potholeCount} ${potholeCount === 1 ? 'pothole' : 'potholes'} • max conf ${maxConfPct}% • ${proximity}`;
  }

  // 2. Driver Status
  let driverTitle = 'MODEL STANDBY';
  let driverSubtext = 'Click "Start Camera" to test';
  let driverColor = 'text-slate-400';

  if (isDriverLive) {
    const reasons = liveDriverState.reason_codes ? ` • ${liveDriverState.reason_codes.slice(0, 2).join(', ')}` : '';
    if (driverState === 'CRITICAL') {
      driverTitle = '🚨 CRITICAL FATIGUE';
      driverColor = 'text-red-400 font-bold animate-pulse';
      driverSubtext = `Closure ${(liveDriverState.driver_state_v2?.metrics?.eye_closure_duration_ms || 0)}ms${reasons}`;
    } else if (driverState === 'DROWSY') {
      driverTitle = 'DROWSINESS DETECTED';
      driverColor = 'text-red-400';
      driverSubtext = `Fatigue ${(driverSmoothed * 100).toFixed(0)}% • ${liveDriverState.eye_state}${reasons}`;
    } else if (driverState === 'FATIGUE_RISK' || driverState === 'LOW_VIGILANCE') {
      driverTitle = 'FATIGUE RISK';
      driverColor = 'text-amber-400';
      driverSubtext = `Fatigue ${(driverSmoothed * 100).toFixed(0)}% • EAR ${liveDriverState.ear.toFixed(2)}${reasons}`;
    } else {
      driverTitle = 'ALERT & ATTENTIVE';
      driverColor = 'text-emerald-400';
      driverSubtext = `EAR ${liveDriverState.ear.toFixed(2)} • Blinks ${liveDriverState.blink_count || 0}`;
    }
  }

  // 3. Fused System Risk
  let systemRisk = 'NORMAL';
  let systemRiskColor = 'text-emerald-400';
  let systemScore = 5;

  let baseHazardScore = road.hazard_score || 0;
  if (isPothole) {
    if (proximity === 'NEAR') baseHazardScore = Math.max(65, baseHazardScore);
    else if (proximity === 'MEDIUM') baseHazardScore = Math.max(40, baseHazardScore);
    else baseHazardScore = Math.max(20, baseHazardScore);
  }

  let driverRiskContribution = 0;
  if (isDriverLive) {
    if (driverState === 'CRITICAL') driverRiskContribution = 100;
    else if (driverState === 'DROWSY') driverRiskContribution = 70;
    else if (driverState === 'FATIGUE_RISK' || driverState === 'LOW_VIGILANCE') driverRiskContribution = 35;
    else driverRiskContribution = Math.round(driverSmoothed * 25);
  }

  systemScore = Math.min(95, Math.max(5, Math.round(baseHazardScore * 0.6 + driverRiskContribution * 0.4)));

  if (systemScore >= 70 || (isPothole && driverState === 'DROWSY')) {
    systemRisk = 'CRITICAL';
    systemRiskColor = 'text-red-400 font-bold';
  } else if (systemScore >= 50 || driverState === 'DROWSY' || (isPothole && proximity === 'NEAR')) {
    systemRisk = 'HIGH';
    systemRiskColor = 'text-red-400';
  } else if (systemScore >= 30 || driverState === 'LOW_VIGILANCE' || isPothole) {
    systemRisk = 'CAUTION';
    systemRiskColor = 'text-amber-400';
  } else {
    systemRisk = 'NORMAL';
    systemRiskColor = 'text-emerald-400';
  }

  // 4. Central Action Warning
  let actionText = 'ROAD CLEAR — DRIVER ALERT — ALL SYSTEMS NORMAL';
  let actionStyle = 'bg-[#0c101a] border-slate-800/80 text-slate-400';

  if (driverState === 'DROWSY' && isPothole) {
    actionText = '🚨 CRITICAL: DROWSY DRIVER & POTHOLE DETECTED AHEAD';
    actionStyle = 'bg-red-950/50 border-red-600 text-red-200 font-bold animate-pulse';
  } else if (driverState === 'DROWSY') {
    actionText = '😴 DRIVER FATIGUE DETECTED — PULL OVER SAFELY';
    actionStyle = 'bg-red-950/40 border-red-700/80 text-red-300 font-bold';
  } else if (driverState === 'LOW_VIGILANCE') {
    actionText = '⚠️ DRIVER INATTENTION DETECTED — MAINTAIN ROAD FOCUS';
    actionStyle = 'bg-amber-950/30 border-amber-700/60 text-amber-300 font-bold';
  } else if (isPothole) {
    if (proximity === 'NEAR') {
      actionText = 'POTHOLE AHEAD — REDUCE SPEED';
      actionStyle = 'bg-red-950/30 border-red-700/60 text-red-300 font-bold';
    } else if (proximity === 'MEDIUM') {
      actionText = 'ROAD HAZARD DETECTED — MAINTAIN CAUTION';
      actionStyle = 'bg-amber-950/30 border-amber-700/60 text-amber-300 font-bold';
    } else {
      actionText = 'POTENTIAL HAZARD DETECTED — STAY ALERT';
      actionStyle = 'bg-amber-950/20 border-amber-800/40 text-amber-400';
    }
  }

  return (
    <div className="min-h-screen bg-[#07090e] text-[#cbd5e1] font-mono flex flex-col justify-between p-6 max-w-[1500px] mx-auto selection:bg-slate-800">
      {/* Hidden processing canvas */}
      <canvas ref={canvasRef} className="hidden" />

      {/* ── HEADER ── */}
      <header className="flex justify-between items-center pb-4 border-b border-slate-800/60">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-lg font-bold tracking-[0.2em] text-white">ROADGUARD AI</h1>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800/60 text-slate-400 tracking-wider">
              PROTOTYPE v0.1
            </span>
          </div>
          <p className="text-[11px] text-slate-500 tracking-widest uppercase mt-0.5">
            Real-Time Driver & Road Safety Intelligence
          </p>
        </div>

        <div className="flex items-center space-x-6">
          <div className="text-right hidden sm:block">
            <div className="text-xs text-slate-300 tracking-wider font-semibold">
              {metrics.fps ? metrics.fps.toFixed(0) : '30'} FPS •{' '}
              {isDriverLive
                ? `${liveDriverState.latency_ms.toFixed(0)} ms (D) / ${(metrics.latency_ms || 30).toFixed(0)} ms (R)`
                : `${metrics.latency_ms ? metrics.latency_ms.toFixed(0) : '36'} ms`}
            </div>
            <div className="text-[10px] text-slate-500 tracking-widest uppercase">Inference Telemetry</div>
          </div>

          <button
            onClick={() => setUseMock(!useMock)}
            className="flex items-center space-x-2 bg-[#0e1320] hover:bg-[#151c2e] px-3 py-1.5 rounded border border-slate-800 transition-colors text-xs"
          >
            <div className={`w-2 h-2 rounded-full ${useMock ? 'bg-amber-500' : 'bg-emerald-400 animate-pulse'}`} />
            <span className="text-slate-300 font-bold tracking-wider">{useMock ? 'MOCK' : 'LIVE'}</span>
          </button>
        </div>
      </header>

      {/* ── HERO PERCEPTION VIEWPORTS (~70% Visual Area) ── */}
      <main className="grid grid-cols-1 md:grid-cols-12 gap-4 my-4 flex-grow items-stretch">
        {/* DRIVER CAMERA (~35% width, md:col-span-4) */}
        <div className="md:col-span-4 bg-[#0a0d17] border border-slate-800/70 rounded-lg overflow-hidden flex flex-col min-h-[380px] lg:min-h-[500px]">
          {/* Viewport Top Bar */}
          <div className="px-3 py-2 bg-[#0e1320]/60 border-b border-slate-800/60 flex justify-between items-center text-[10px] tracking-wider text-slate-400">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${isCameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
              <span className="text-slate-300 font-semibold">DRIVER CAMERA</span>
              <span className="text-slate-500">• PyTorch CNN</span>
            </div>

            <div className="flex items-center space-x-2">
              {isCameraActive ? (
                <button
                  onClick={stopCamera}
                  className="px-2 py-0.5 text-[10px] font-bold rounded bg-red-950/60 border border-red-700/60 text-red-300 hover:bg-red-900/80 transition-colors"
                >
                  STOP CAM
                </button>
              ) : (
                <button
                  onClick={startCamera}
                  className="px-2 py-0.5 text-[10px] font-bold rounded bg-emerald-950/60 border border-emerald-700/60 text-emerald-300 hover:bg-emerald-900/80 transition-colors"
                >
                  START CAM
                </button>
              )}
            </div>
          </div>

          {/* Viewport Center: Live Video or Standby State */}
          <div className="flex-grow relative bg-black flex items-center justify-center overflow-hidden">
            {/* Real Webcam Element */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover -scale-x-100 ${isCameraActive ? 'block' : 'hidden'}`}
            />

            {/* Standby Placeholder */}
            {!isCameraActive && (
              <div className="flex-grow flex flex-col items-center justify-center p-6 text-center relative bg-[radial-gradient(#151d30_1px,transparent_1px)] [background-size:16px_16px]">
                <div className="w-14 h-14 rounded-full border border-dashed border-slate-700 flex items-center justify-center mb-3 text-slate-500">
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <div className="text-xs font-bold tracking-widest text-slate-300 uppercase">DRIVER CAMERA STANDBY</div>
                <div className="text-[10px] text-slate-500 tracking-wider mt-1 max-w-[240px] mb-4">
                  PyTorch CustomDriverCNN model ready for live face & drowsiness testing
                </div>

                <button
                  onClick={startCamera}
                  className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-bold text-xs tracking-wider transition-all shadow-lg shadow-emerald-950/50 flex items-center space-x-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>TEST WITH WEBCAM</span>
                </button>

                {cameraError && (
                  <div className="text-[10px] text-red-400 bg-red-950/40 border border-red-800/40 px-2 py-1 rounded mt-3 max-w-[240px]">
                    {cameraError}
                  </div>
                )}
              </div>
            )}

            {/* Live Camera HUD Telemetry Overlay */}
            {isCameraActive && (
              <>
                {/* Top Status Pill */}
                <div className="absolute top-3 left-3 flex items-center space-x-2">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider uppercase border shadow-md ${
                      driverState === 'CRITICAL'
                        ? 'bg-red-950/95 border-red-500 text-red-100 font-extrabold animate-pulse'
                        : driverState === 'DROWSY'
                        ? 'bg-red-950/90 border-red-500 text-red-200'
                        : driverState === 'FATIGUE_RISK' || driverState === 'LOW_VIGILANCE'
                        ? 'bg-amber-950/90 border-amber-500 text-amber-200'
                        : 'bg-emerald-950/90 border-emerald-500 text-emerald-200'
                    }`}
                  >
                    {driverState === 'CRITICAL'
                      ? '🚨 CRITICAL'
                      : driverState === 'DROWSY'
                      ? '🔴 DROWSY'
                      : driverState === 'FATIGUE_RISK' || driverState === 'LOW_VIGILANCE'
                      ? '🟡 FATIGUE RISK'
                      : '🟢 NORMAL'}
                  </span>
                </div>

                {/* Bottom Telemetry Bar */}
                <div className="absolute bottom-2 left-2 right-2 bg-[#07090e]/95 backdrop-blur-md border border-slate-800/80 rounded p-2 text-[10px] space-y-1.5 shadow-xl">
                  <div className="flex justify-between items-center text-slate-400">
                    <div className="flex items-center space-x-1.5">
                      <span className="font-semibold text-slate-300">FATIGUE RISK:</span>
                      <span className="text-white font-bold text-xs">{((liveDriverState?.fatigue_risk_score ?? driverSmoothed) * 100).toFixed(0)}%</span>
                    </div>
                    {liveDriverState?.signal_disagreement && (
                      <span className="px-1.5 py-0.2 text-[8px] bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded font-semibold animate-pulse">
                        DISAGREEMENT
                      </span>
                    )}
                  </div>
                  {/* Fatigue Risk Bar */}
                  <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-200 ${
                        (liveDriverState?.fatigue_risk_score ?? driverSmoothed) > 0.65
                          ? 'bg-red-500'
                          : (liveDriverState?.fatigue_risk_score ?? driverSmoothed) > 0.35
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.min(100, Math.max(2, (liveDriverState?.fatigue_risk_score ?? driverSmoothed) * 100))}%` }}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[9px] text-slate-400 pt-1 border-t border-slate-800/60">
                    <div>
                      CNN Drowsy: <strong className="text-slate-200">{((liveDriverState?.cnn_drowsy_probability ?? driverProb) * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="text-right">
                      Eye: <strong className="text-slate-200">{liveDriverState?.eye_state || 'OPEN'}</strong> (EAR {liveDriverState?.ear.toFixed(2) || '0.30'})
                    </div>
                    <div>
                      PERCLOS: <strong className="text-slate-200">{((liveDriverState?.driver_state_v2?.metrics?.perclos ?? 0) * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="text-right">
                      Recent Blinks: <strong className="text-slate-200">{liveDriverState?.recent_signals?.long_blinks_30s ?? 0}</strong>
                    </div>
                    <div>
                      Active Yawn: <strong className="text-slate-200">{liveDriverState?.active_signals?.active_yawn ? 'YES' : 'NO'}</strong>
                    </div>
                    <div className="text-right">
                      Tracking: <strong className="text-slate-200">{liveDriverState?.tracking_state || (liveDriverState?.face_detected ? 'TRACKED' : 'LOST')}</strong>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        {/* ROAD CAMERA (~65% width, md:col-span-8) — HERO! */}
        <div className="md:col-span-8 bg-[#0a0d17] border border-slate-800/70 rounded-lg overflow-hidden flex flex-col min-h-[380px] lg:min-h-[500px]">
          {/* Viewport Top Bar */}
          <div className="px-3 py-2 bg-[#0e1320]/60 border-b border-slate-800/60 flex justify-between items-center text-[10px] tracking-wider text-slate-400">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-slate-300 font-semibold">ROAD CAMERA</span>
              <span className="text-slate-500">• YOLOv8n PERCEPTION</span>
            </div>
            <span className="text-emerald-400 font-semibold bg-emerald-950/40 px-1.5 py-0.5 rounded border border-emerald-800/40">
              REAL MODEL LIVE
            </span>
          </div>

          {/* Live Video Feed with Real YOLO Detections */}
          <div className="flex-grow relative bg-black flex items-center justify-center overflow-hidden">
            {road.frame_data ? (
              <img src={road.frame_data} alt="Live Road Inference Feed" className="w-full h-full object-cover" />
            ) : (
              <div className="text-center p-8 space-y-2">
                <div className="text-xs tracking-widest text-slate-500 uppercase">Awaiting Road Inference Stream...</div>
                <div className="text-[10px] text-slate-600">Verify backend process is active on localhost:8000</div>
              </div>
            )}

            {/* Corner telemetry tag */}
            {isPothole && (
              <div className="absolute top-3 right-3 bg-red-950/80 border border-red-600/70 text-red-300 px-2.5 py-1 rounded text-[11px] font-bold tracking-wider shadow-lg">
                HAZARD DETECTED
              </div>
            )}
          </div>
        </div>
      </main>

      {/* ── MINIMAL HORIZONTAL INTELLIGENCE STRIP (3 Columns) ── */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6 py-4 px-2 border-t border-b border-slate-800/70 text-left">
        {/* DRIVER */}
        <div className="space-y-1">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">DRIVER VIGILANCE</div>
          <div className={`text-sm font-semibold tracking-wider ${driverColor}`}>{driverTitle}</div>
          <div className="text-[11px] text-slate-400">{driverSubtext}</div>
        </div>

        {/* ROAD */}
        <div className="space-y-1">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">ROAD PERCEPTION</div>
          <div className={`text-sm font-semibold tracking-wider ${roadColor}`}>{roadTitle}</div>
          <div className="text-[11px] text-slate-400">{roadSubtext}</div>
        </div>

        {/* SYSTEM */}
        <div className="space-y-1">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">SAFETY INDEX</div>
          <div className={`text-sm font-semibold tracking-wider ${systemRiskColor}`}>{systemRisk}</div>
          <div className="text-[11px] text-slate-400">{systemScore}/100 Risk Index</div>
        </div>
      </section>

      {/* ── FULL-WIDTH ADAS ACTION / WARNING BAR ── */}
      <footer className="mt-4 space-y-3">
        <div className={`w-full py-3.5 px-4 rounded border text-center text-xs tracking-[0.2em] transition-all duration-300 ${actionStyle}`}>
          {actionText}
        </div>

        {/* Diagnostics link toggle */}
        <div className="flex justify-between items-center text-[10px] text-slate-600 px-1 pt-1">
          <span>ROADGUARD AI • DUAL-PERCEPTION ACTIVE</span>
          <button
            onClick={() => setShowDiagnostics(!showDiagnostics)}
            className="hover:text-slate-400 transition-colors uppercase tracking-wider underline underline-offset-4"
          >
            {showDiagnostics ? '[ Close Diagnostics ]' : '[ Engineering Telemetry & Diagnostics ]'}
          </button>
        </div>
      </footer>

      {/* Optional Collapsed Diagnostics Drawer */}
      {showDiagnostics && (
        <div className="mt-6">
          <DiagnosticsView data={current.data} connected={current.connected} reconnecting={current.reconnecting} />
        </div>
      )}
    </div>
  );
}

export default App;
