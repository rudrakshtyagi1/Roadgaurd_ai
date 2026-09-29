# 🛡️ RoadGuard AI — Next-Gen Dual-Perception ADAS & Road Safety Platform

[![Tests Passing](https://img.shields.io/badge/Tests-96%2F96%20Passing-brightgreen?style=for-the-badge&logo=pytest)](ARCHITECTURE.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-3178C6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38B2AC?style=for-the-badge&logo=tailwind-css)](https://tailwindcss.com)

> **RoadGuard AI** is a real-time, dual-perception Advanced Driver Assistance System (ADAS) and road safety platform. By fusing **cabin driver monitoring (DMS)** with **forward road hazard and traffic perception**, RoadGuard AI solves the fatal blind spots of single-domain safety systems through explainable, temporal multi-hazard fusion and priority arbitration.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Master System Architecture](#-master-system-architecture)
- [Key Features & Capabilities](#-key-features--capabilities)
- [Engineering Phases & Verification](#-engineering-phases--verification)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start & Installation](#-quick-start--installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#1-backend-setup)
  - [Frontend Setup](#2-frontend-setup)
  - [Running the Complete Stack](#3-running-the-complete-stack)
- [Automated Test Suite](#-automated-test-suite)
- [API Reference](#-api-reference)
  - [Video Inference](#video-inference)
  - [Crowdsourced Hazard Network](#crowdsourced-hazard-network)
  - [Safer Route Engine](#safer-route-engine)
  - [WebSocket Telemetry](#websocket-telemetry)
- [Edge Deployment & Profiling](#-edge-deployment--profiling)
- [Contributing & License](#-license)

---

## 🌟 Overview

Conventional vehicle safety solutions are siloed:
* **Cabin-only systems** detect if a driver is nodding off, but have zero awareness of an obstacle or crater 15 meters ahead.
* **Forward-only collision systems** warn about obstacles without knowing if the driver is already braking or completely unconscious.

**RoadGuard AI bridges this divide.** It combines deep facial feature extraction, real-time road object detection, perspective ego-lane corridor tracking, and image-space Time-To-Collision (TTC) with a centralized **RoadGuard Brain**. When a driver is fatigued, reaction times slow down—so RoadGuard automatically amplifies road hazard sensitivity, arbitrates audio alerts to prevent driver panic, and logs geotagged hazards to a decentralized safety network.

---

## 🏛️ Master System Architecture

```
                                  ROADGUARD AI ARCHITECTURE
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         DRIVER CABIN CAMERA (DMS)                       FORWARD ROAD CAMERA (ADAS)
                      │                                               │
                      ▼                                               ▼
            Phase 1: Driver AI                        Phase 2: Traffic & Phase 3: Road
    ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
    │  • MediaPipe FaceLandmarker     │             │  • YOLOv8 Vehicle & Pedestrian  │
    │  • EAR / MAR Computation        │             │  • YOLOv8 Pothole & Defect Net  │
    │  • 3D Head Pose (Yaw/Pitch/Roll)│             │  • Perspective Ego-Corridor     │
    │  • CustomDriverCNN (PyTorch)    │             │  • IoU Tracker & Track State    │
    │  • PERCLOS & Monotonic Engine   │             │  • Rolling Time-To-Collision    │
    └────────────────┬────────────────┘             └────────────────┬────────────────┘
                     │                                               │
                     ▼                                               ▼
              DRIVER RISK STATE                              FORWARD RISK STATE
             (Vigilance / Fatigue)                        (Collision / Road Hazard)
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                                  ROADGUARD BRAIN (Phase 4)
                           Unified Temporal Multi-Hazard Fusion
                        Contextual Gating & Compound Risk Analysis
                                             │
                                             ▼
                                 DECISION ENGINE (Phase 5)
                           Priority Arbitration & Audio De-Spam
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
            DRIVER INTERVENTION                             CONNECTED NETWORK (Phase 6)
       • Visual Heads-Up Display                      • Spatial Hazard Clustering (DBSCAN)
       • Prioritized Audio / Voice                    • Crowd-Sourced Hazard DB
       • Haptic Pulses                                • Safety-Weighted Route Scoring
                                                                    │
                                                                    ▼
                                                       EDGE AI & FLYWHEEL (Phases 7 & 8)
                                                      • 30 FPS Dynamic Pipeline Scheduler
                                                      • Load Shedding & Resource Profiler
                                                      • Hard Case Auto-Miner
                                                      • Multi-Component Ablation Engine
```

---

## 🚀 Key Features & Capabilities

### 1. Driver Intelligence & Vigilance (DMS)
* **Custom PyTorch CNN**: Trained deep facial drowsiness classifier delivering calibrated probabilities.
* **Geometric Facial Landmarks**: 478-point MediaPipe mesh calculating Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR for yawns), and 3D Head Pose (Yaw, Pitch, Roll).
* **DriverStateEngine V2**: Monotonic temporal tracking with PERCLOS (percentage of eyelid closure over time), blink duration hysteresis, and clear reason coding (`EYES_CLOSED_PROLONGED`, `FREQUENT_YAWNING`, `MICRO_SLEEP`).

### 2. Forward Collision Intelligence (ADAS)
* **YOLOv8 Detection**: High-speed identification of vehicles, trucks, motorcycles, and pedestrians.
* **Perspective Ego-Corridor**: Trapeze-shaped dynamic driving lane projection to filter out non-threatening adjacent-lane traffic.
* **Rolling-Window Time-To-Collision (TTC)**: Scale-rate bounding-box expansion calculations providing sub-second collision warnings without requiring costly LiDAR hardware.

### 3. Road Surface Hazard Intelligence
* **Trained YOLOv8 Pothole Detector**: Specialized single-class weights trained to identify road depressions, craters, and surface breakages in diverse lighting conditions.
* **Spatial Hazard Tracker**: Tracks hazard persistence across consecutive frames to eliminate transient camera artifacts and flickering.
* **Severity & Distance Estimator**: Calibrated ground-plane geometry estimating distance in meters and classifying hazards (`NEAR`, `MEDIUM`, `FAR`, `CRITICAL`).

### 4. RoadGuard Brain (Unified Temporal Fusion)
* **Contextual Risk Gating**: Dynamically scales hazard urgency based on driver status. A pothole ahead while a driver is attentive triggers a mild advisory; the same pothole when the driver is drowsy triggers a high-priority audible intervention.
* **Explainable Reason Coding**: Generates verifiable diagnostic tags for every decision (e.g. `FATIGUED_DRIVER_APPROACHING_HAZARD`, `IMMINENT_COLLISION_TAKEOVER`).

### 5. Intelligent Alert & Intervention Engine
* **Strict Priority Hierarchy**: `CRITICAL COLLISION` > `CRITICAL ROAD HAZARD` > `DRIVER DROWSINESS` > `CAUTION ADVISORY`.
* **Anti-Spam & Audio Exclusivity**: Enforces 2.5-second hysteresis debouncing to prevent ear fatigue and alarming auditory ping storms.

### 6. Connected Road Network & Safer Routing
* **Crowdsourced Hazard DB**: Stores and aggregates geotagged reports with confidence decays.
* **DBSCAN Spatial Clustering**: Merges repeated detections of the same pothole from multiple passes or vehicles.
* **Safer Route Engine**: Evaluates alternative GPS routes by road quality index, steering drivers away from high-density hazard corridors.

### 7. Edge AI & Data Flywheel
* **Adaptive Multi-Rate Scheduler**: Runs driver landmarks at 30 Hz, vehicle detection at 15 Hz, and pothole detection at 10 Hz with tracking running every frame. Automatically activates load shedding if latency exceeds 33 ms.
* **Hard Case Auto-Mining**: Detects ambiguous predictions (confidence 0.25–0.40), tracking losses, and model-physiology disagreements for automated dataset curation.

### 8. Cockpit Dashboard & Diagnostics
* **Live Dual-Feed HUD**: Streamlined, cockpit-ready React interface displaying live webcam input and forward camera feeds with HUD telemetry overlays.
* **Engineering Diagnostics Drawer**: Real-time breakdown of Euler angles, EAR, MAR, PERCLOS, raw bounding box confidences, WebSocket status, and device acceleration.

---

## 📊 Engineering Phases & Verification

RoadGuard AI was built with rigorous test-driven architecture across 9 engineering phases:

| Phase | Subsystem | Core Module | Verification Status |
|---|---|---|---|
| **Phase 0** | Core Foundation | `PerceptionEvent`, `PipelineRunner`, `TemporalBuffer` | ✅ **Passed** |
| **Phase 1** | Driver Intelligence | `DriverMonitor`, `DriverStateEngineV2`, `DriverClassifier` | ✅ **Passed** (20 Scenarios) |
| **Phase 2 & 2.5** | Collision Intelligence | `CollisionIntelligenceEngine`, `PerspectiveEgoCorridor`, `TTCEstimator` | ✅ **Passed** (14 Scenarios) |
| **Phase 3** | Road Hazard Intelligence | `RoadHazardEngine`, `HazardTracker`, `SeverityEstimator` | ✅ **Passed** |
| **Phase 4** | RoadGuard Brain | `RoadGuardBrain` (Unified Temporal Fusion) | ✅ **Passed** |
| **Phase 5** | Intelligent Decision | `AlertPrioritizerEngine` (Audio Arbitration & Anti-Spam) | ✅ **Passed** |
| **Phase 6** | Road Intelligence Network | `HazardDatabase`, `SaferRouteEngine`, REST API | ✅ **Passed** |
| **Phase 7** | Edge AI Optimization | `AdaptivePipelineScheduler`, `EdgeProfiler` | ✅ **Passed** |
| **Phase 8** | Data Flywheel | `HardCaseMiner`, `AblationSuite` | ✅ **Passed** |

> **Automated Test Discovery**: **96 unit and benchmark tests passing (0 failures, 0 errors)** in ~1.3 seconds.

---

## 💻 Tech Stack

### Core AI & Computer Vision
* **PyTorch (`torch`, `torchvision`)**: Deep neural network runtime for `CustomDriverCNN`.
* **Ultralytics YOLOv8**: Real-time object and pothole detection.
* **Google MediaPipe**: Fast 478-point facial landmark topology on CPU/MPS.
* **OpenCV (`cv2`)**: Frame decoding, matrix operations, perspective transformations.
* **NumPy & Scikit-learn**: Spatial geometry, rolling statistics, and DBSCAN clustering.

### Backend & Streaming
* **FastAPI**: Modern, asynchronous REST API with Pydantic v2 schemas.
* **WebSockets**: Bi-directional 30 FPS state and event streaming.
* **Uvicorn**: High-performance ASGI web server.
* **Apple Silicon MPS / CUDA / CPU**: Hardware acceleration auto-detection.

### Cockpit Frontend
* **React 18 & TypeScript**: Robust, type-safe reactive dashboard.
* **Vite**: Rapid HMR development and optimized production bundling.
* **Tailwind CSS**: Automotive dark-mode aesthetic designed for low glare.
* **Lucide Icons & Recharts**: Clean visualization components.

---

## 📁 Project Directory Structure

```plaintext
roadgaurdai/
├── ARCHITECTURE.md                  # Comprehensive architectural specification
├── README.md                        # Project documentation (this file)
├── backend/
│   ├── main.py                      # FastAPI app entry point & WebSocket server
│   ├── requirements.txt             # Python dependencies
│   ├── configs/                     # Engine YAML parameter files
│   ├── model_assets/                # Neural network weights
│   │   ├── best_driver_model.pt     # CustomDriverCNN PyTorch weights
│   │   ├── best_pothole.pt          # Trained YOLOv8 pothole model
│   │   └── face_landmarker.task     # MediaPipe FaceLandmarker bundle
│   ├── core/                        # Core perception & reasoning engines
│   │   ├── brain/                   # RoadGuard Brain (temporal fusion)
│   │   ├── collision/               # YOLOv8 ADAS, IoU tracker & TTC
│   │   ├── decision/                # Alert prioritizer & audio debouncing
│   │   ├── driver_monitor.py        # DMS pipeline (MediaPipe + CNN)
│   │   ├── driver_state_engine.py   # Hysteretic driver state machine V2
│   │   ├── edge/                    # Adaptive scheduler & profiler
│   │   ├── flywheel/                # Hard case auto-miner & ablations
│   │   ├── hazard/                  # Pothole tracker & ego-lane corridor
│   │   └── network/                 # Geotagged hazard DB & routing
│   ├── routers/                     # REST API endpoints (video, hazards, map)
│   ├── tests/                       # 96 automated unit & benchmark tests
│   └── ws/                          # WebSocket streaming & inference loop
├── frontend/
│   ├── index.html                   # Web application entry
│   ├── package.json                 # Node dependencies & scripts
│   ├── vite.config.ts               # Vite configuration
│   └── src/
│       ├── App.tsx                  # Main cockpit HUD & webcam integration
│       ├── components/              # DiagnosticsView & UI widgets
│       └── hooks/                   # WebSocket and mock data hooks
├── potholes/                        # Sample road videos for simulation
└── scratch/                         # Benchmarking & verification scripts
```

---

## ⚡ Quick Start & Installation

### Prerequisites
* **Python 3.10+**
* **Node.js 18+** and **npm**
* A webcam (for live driver monitoring) or recorded test clips

---

### 1. Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify model assets**:
   Ensure `best_driver_model.pt`, `best_pothole.pt`, and `face_landmarker.task` are present in `backend/model_assets/`.

5. **Start the backend server**:
   ```bash
   python main.py
   ```
   * The API server starts at `http://localhost:8000`.
   * Interactive API docs (Swagger): `http://localhost:8000/docs`.
   * Real-time WebSocket: `ws://localhost:8000/ws`.

---

### 2. Frontend Setup

1. **Open a new terminal and navigate to the frontend**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the Vite development server**:
   ```bash
   npm run dev
   ```
   * The dashboard opens at `http://localhost:5173`.

---

### 3. Running the Complete Stack

1. Open `http://localhost:5173` in your browser.
2. Click **"START CAM"** on the Driver Camera panel to test live drowsiness detection with your webcam.
3. The dashboard will automatically stream forward road perception from the backend, computing fused risk levels in real time.
4. Click **"[ Engineering Telemetry & Diagnostics ]"** at the bottom to view live EAR, MAR, PERCLOS, head pose Euler angles, and detection stats.

---

## 🧪 Automated Test Suite

RoadGuard AI includes a full test suite validating all phases of the pipeline:

```bash
cd backend
python -m unittest discover -s tests -v
```

### Test Scope:
* `test_driver_state_engine.py`: Validates blink counts, EAR decay, and hysteresis.
* `test_driver_classifier.py`: Tests PyTorch CustomDriverCNN forward pass and tensor shapes.
* `test_collision_tracker.py` & `test_ttc.py`: Validates IoU multi-object tracking and Time-To-Collision formulas.
* `test_hazard_engine.py`: Verifies pothole spatial tracking and ego-path relevance.
* `test_brain.py`: Tests unified risk arbitration under compounded hazards.
* `test_decision_engine.py`: Validates audio debouncing and alert priority hierarchies.
* `test_edge_and_flywheel.py`: Tests adaptive load shedding and hard-case mining.

---

## 📡 API Reference

### Video Inference
* `POST /api/video/driver`: Accepts a single multipart JPEG/PNG frame and returns complete driver telemetry (EAR, MAR, Head Pose, CNN probability, and state).
* `POST /api/video/road`: Accepts a forward camera frame and returns detected road hazards, distance estimates, and bounding boxes.
* `POST /api/inference/`: Performs simultaneous dual-frame perception, temporal buffering, and unified risk calculation.

### Crowdsourced Hazard Network
* `GET /api/hazards/`: Fetches all clustered road hazards with coordinates and severity ratings.
* `POST /api/hazards/report`: Ingests a new hazard event report:
  ```json
  {
    "hazard_type": "pothole",
    "severity": "HIGH",
    "confidence": 0.92,
    "latitude": 28.7041,
    "longitude": 77.1025,
    "vehicle_id": "car_alpha_01"
  }
  ```

### Safer Route Engine
* `POST /api/map/safer_route`: Evaluates alternative route coordinates against hazard cluster densities:
  ```json
  {
    "start_lat": 28.7041,
    "start_lon": 77.1025,
    "dest_lat": 28.5355,
    "dest_lon": 77.3910
  }
  ```

### WebSocket Telemetry
* `ws://localhost:8000/ws`: High-frequency binary/JSON stream delivering continuous updates to the cockpit HUD, including metrics, FPS, driver state, road hazards, and central warning directives.

---

## ⚡ Edge Deployment & Profiling

RoadGuard AI is optimized for resource-constrained edge hardware (Apple Silicon MPS, NVIDIA Jetson, Intel NUC):

* **Multi-Rate Scheduling**:
  * Face Mesh & Head Pose: 30 Hz
  * Forward Obstacle Tracking: 30 Hz
  * Deep Forward Detection: 15 Hz
  * Pothole YOLO Inference: 10 Hz
* **Dynamic Load Shedding**: If p95 frame latency exceeds 33 ms, secondary detector cadences are automatically down-throttled to preserve driver safety loop responsiveness.

---

## 🛡️ License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

---

<p align="center">
  <b>Built with precision for safer roads and intelligent cockpits.</b><br>
  <sub>RoadGuard AI Engineering Team</sub>
</p>