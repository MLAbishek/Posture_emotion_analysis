# 📋 Project Report & PPT Content Guide
## AI-Powered Posture & Emotion Analysis System

---

# ═══════════════════════════════════════════
# PART 1: COMPLETE PROJECT INFORMATION
# ═══════════════════════════════════════════

## 🔷 Project Title
**AI-Powered Real-Time Posture and Emotion Analysis System for Workplace Wellness**

---

## 🔷 One-Line Summary
A real-time computer vision system that monitors a user's posture, fatigue level, and emotional state through a webcam, logs session data to a database, and uses AI agents (LLM + RAG + SQL) to provide actionable workplace wellness insights.

---

## 🔷 Problem Statement
Modern desk workers spend long hours in front of computers, often developing:
- **Poor posture** (forward head, slouching) leading to musculoskeletal issues
- **Fatigue & drowsiness** reducing productivity and safety
- **Stress and negative emotional states** going undetected

Traditional wellness solutions require manual input or wearable hardware. This project proposes a **non-intrusive, camera-based AI system** that passively monitors and alerts users in real-time.

---

## 🔷 Objectives
1. Detect and alert for **bad posture** (neck angle deviation from calibrated baseline)
2. Detect **drowsiness/fatigue** using Eye Aspect Ratio (EAR)
3. Recognize **basic emotions** (Happy, Sad, Surprised, Neutral) via facial blendshapes
4. **Log all session data** to a PostgreSQL database for historical analysis
5. Provide **AI-powered insights** by combining SQL analytics with a RAG knowledge base
6. Display a **real-time dashboard overlay** on the webcam feed

---

## 🔷 Tech Stack (Complete)

| Category | Technology | Purpose |
|---|---|---|
| **Computer Vision** | OpenCV (`cv2`) | Camera capture, image processing, dashboard rendering |
| **Pose Estimation** | MediaPipe Pose Landmarker | 33-point body skeleton detection |
| **Face Analysis** | MediaPipe Face Landmarker | 468 facial landmark + blendshape detection |
| **Numerical Computing** | NumPy | Angle and distance calculations |
| **Database ORM** | SQLAlchemy | Database models and session management |
| **Database** | PostgreSQL | Persistent session and log storage |
| **DB Driver** | psycopg2-binary | PostgreSQL Python adapter |
| **LLM** | LLaMA 3.3 70B (via Groq) | Natural language agent reasoning |
| **SQL Agent** | LangChain SQL Agent | Natural language → SQL query generation |
| **RAG Agent** | LangChain + Pinecone | Knowledge retrieval from wellness PDFs |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Text embedding for vector search |
| **Vector Store** | Pinecone | Cloud vector database for knowledge base |
| **Agent Orchestration** | LangChain `RunnableParallel` | Parallel SQL + RAG execution |
| **Config Management** | python-dotenv | Secure API key loading |

---

## 🔷 System Architecture (Describe This in Your Diagram)

```
┌─────────────────────────────────────────────────────────────┐
│                        WEBCAM INPUT                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │        main.py          │  ← Entry point
              │   (Main Control Loop)   │
              └─┬──────────┬────────────┘
                │          │
    ┌───────────▼──┐   ┌───▼──────────────┐
    │ PoseDetector │   │  FaceAnalyzer     │
    │ (MediaPipe   │   │  (MediaPipe Face  │
    │  Pose)       │   │   Landmarker)     │
    └───────┬──────┘   └──────┬────────────┘
            │                 │
            │         ┌───────▼────────────┐
            │         │  Fatigue Detection  │  ← EAR Algorithm
            │         │  Emotion Detection  │  ← Blendshape Logic
            │         └───────┬────────────┘
            │                 │
    ┌───────▼─────────────────▼────┐
    │       Visualizer              │  ← Real-time Dashboard Overlay
    └──────────────┬────────────────┘
                   │
    ┌──────────────▼────────────────┐
    │       DataUploader             │  ← SQLAlchemy ORM
    └──────────────┬────────────────┘
                   │
    ┌──────────────▼────────────────┐
    │     PostgreSQL Database        │
    │  sessions + logdetails tables  │
    └──────────────┬────────────────┘
                   │
    ┌──────────────▼────────────────────────────────┐
    │              AI Agent Layer                    │
    │                                                │
    │  ┌─────────────┐     ┌──────────────────────┐  │
    │  │  SQL Agent  │     │      RAG Agent       │  │
    │  │ (LangChain) │     │ (Pinecone + LLaMA 3) │  │
    │  └──────┬──────┘     └──────────┬───────────┘  │
    │         └──────────┬────────────┘              │
    │              ┌─────▼─────┐                       │
    │              │ MainAgent │  ← Combines + Responds│
    │              └───────────┘                       │
    └───────────────────────────────────────────────────┘
```

---

## 🔷 Module-Wise Breakdown

### 1. `config.py` — Global Configuration
- Centralized constants for thresholds, camera settings, model paths, colors
- Key constants:
  - `EAR_THRESHOLD = 0.15` → Eye Aspect Ratio below this = drowsy
  - `EAR_CONSECUTIVE_FRAMES = 20` → Must persist for 20 frames to trigger alert
  - `SLOUCH_ANGLE_THRESHOLD = 160°` → Default baseline neck angle
  - `SLOUCH_THRESHOLD_MARGIN = 15°` → How far below baseline triggers alert
  - `NECK_INCLINATION_THRESHOLD = 40°` → Forward head posture detection
  - Camera: 1280×720 resolution

---

### 2. `detectors/pose_detector.py` — Posture Detection Module
**Class: `PoseDetector`**

- Uses **MediaPipe Pose Landmarker** (`pose_landmarker_full.task` model)
- Extracts key skeletal landmarks:
  - Ears (7, 8), Shoulders (11, 12), Hips (23, 24), Nose (0)
- **Calibration System**: User presses 'C' to set their personal baseline angle
- **`calculate_posture_metrics()`**: Returns neck angle, baseline, calibration status, landmark coords
- **Algorithm**:
  - Computes `neck_angle_left = angle(left_ear, left_shoulder, left_hip)`
  - Computes `neck_angle_right = angle(right_ear, right_shoulder, right_hip)`
  - `avg_neck_angle = (left + right) / 2`
  - If `avg_neck_angle < (baseline - 15°)` → **SLOUCHING ALERT**

---

### 3. `detectors/face_analyzer.py` — Fatigue & Emotion Module
**Class: `FaceAnalyzer`**

**A. Fatigue Detection (Eye Aspect Ratio — EAR Algorithm)**
- Uses 6 facial landmarks per eye (Dlib-inspired, adapted for MediaPipe)
- `EAR = (vertical_1 + vertical_2) / (2 × horizontal)`
- **Left Eye landmarks**: [33, 160, 158, 133, 153, 144]
- **Right Eye landmarks**: [362, 385, 387, 263, 373, 380]
- EAR drops sharply when eyes close → sustained drop = drowsy

**B. Emotion Detection (Blendshape Analysis)**
- Uses MediaPipe Face Blendshapes (52 facial action units)
- Emotion Rules:
  - **Happiness**: avg of `mouthSmileLeft` + `mouthSmileRight`
  - **Sadness**: weighted avg of mouth frown + inner brow raise
  - **Surprise**: `browOuterUpLeft/Right` + `jawOpen` (jawOpen must be > 0.25)
  - **Neutral**: if max confidence < 0.2 threshold
- Returns: dominant emotion + confidence score + all scores dict

---

### 4. `utils/geometry.py` — Mathematical Utilities
- `calculate_angle(a, b, c)` — 3-point angle via `arctan2`, returns degrees
- `euclidean_distance(p1, p2)` — Standard Euclidean distance for EAR

---

### 5. `utils/visualizer.py` — Real-Time Dashboard Overlay
**Class: `Visualizer`**

- **Calibration Mode**: Shows "CALIBRATION MODE" banner + current angle
- **Live Mode Dashboard** (semi-transparent top bar):
  - Neck Angle display (GREEN = good, RED = alert)
  - EAR value display
  - Dominant Emotion + confidence (color-coded by emotion)
  - Alerts: "POOR POSTURE!", "DROWSINESS ALERT!" in RED
  - FPS counter
- Draws posture line from ear to shoulder on the video frame

---

### 6. `utils/DataUploader.py` — Database Logging Module
**Class: `DataUploader`**

**Database Schema:**
```
Table: sessions
  - id (PK)
  - start_time (DateTime)
  - end_time (DateTime)

Table: logdetails
  - id (PK)
  - session_id (FK → sessions.id)
  - timestamp (DateTime)
  - neck_angle (Float)
  - is_slouching (Boolean)
  - ear_value (Float)
  - is_drowsy (Boolean)
  - emotion (String)
  - emotion_confidence (Float)
```

- Logs data every **30 frames** (~1 second) to avoid overwhelming the DB
- Manages session lifecycle: `start_session()` → `log_data()` → `end_session()`

---

### 7. `Agents/SQLAgent.py` — SQL Analytics Agent
- Uses **LangChain SQL Agent** with **LLaMA 3.3 70B** (Groq)
- Connects to the PostgreSQL database via `DATABASE_URL`
- Can answer natural language questions about the logged data:
  - *"How many times was I drowsy today?"*
  - *"What was my average neck angle last session?"*
- Read-only: DML statements (INSERT/UPDATE/DELETE) are blocked
- Uses `handle_parsing_errors=True` for robustness

---

### 8. `Agents/RAGAgent.py` — Knowledge Retrieval Agent
- Uses **Pinecone** vector database (`workplace-knowledge` index)
- Knowledge base PDFs ingested:
  - `Ergonomics.pdf`, `Fatigue.pdf`, `Productivity.pdf`, `Workplace_enhancement.pdf`
- Uses **HuggingFace `all-MiniLM-L6-v2`** for embedding
- Retrieves top-4 most relevant document chunks
- Answers questions based on workplace wellness research literature

---

### 9. `Agents/MainAgent.py` — Orchestrator Agent
- Uses **LangChain `RunnableParallel`** to run SQL Agent + RAG Agent **simultaneously**
- Merges both outputs and passes to LLaMA 3.3 70B for a **combined final insight**
- Example query: *"I am slouching too much, how to elucidate it?"*
  - SQL Agent fetches: your slouching frequency/angle data from DB
  - RAG Agent fetches: ergonomics advice from knowledge base
  - Main Agent: synthesizes both into personalized advice

---

### 10. `main.py` — Application Entry Point
**Main Event Loop:**
1. Initialize camera (1280×720)
2. Initialize PoseDetector, FaceAnalyzer, Visualizer, DataUploader
3. **Phase 1 — Calibration**: Display UI, wait for user to press 'C'
4. **Phase 2 — Live Analysis Loop**:
   - Capture frame → Pose Detection → Posture Metrics
   - Face Detection → EAR + Emotion
   - Drowsiness logic (frame counter)
   - Log data to DB (every 30 frames)
   - Draw dashboard overlay
   - Display frame
5. On 'Q': End session, release camera, close windows

---

## 🔷 Key Algorithms

### Eye Aspect Ratio (EAR)
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```
- p1-p6 are eye corner and lid landmark coordinates
- Normal open eye: EAR ≈ 0.25–0.35
- Closed eye: EAR < 0.15
- Drowsiness confirmed after 20 consecutive frames below threshold

### Neck Angle Calculation
```
neck_angle = angle(ear_point, shoulder_point, hip_point)
```
- Measured bilaterally (both sides) and averaged
- Baseline set during calibration (personalized)
- Alert if: `current_angle < baseline - 15°`

### Emotion Detection via Blendshapes
- MediaPipe provides 52 blendshape coefficients (0–1 range)
- Rule-based mapping of blendshapes to emotions
- Confidence threshold (0.2) filters noise → classifies as Neutral

---

## 🔷 Data Flow Summary

```
Camera Frame
    → MediaPipe Pose → Neck Angle → Slouch Check
    → MediaPipe Face → EAR → Drowsy Check
                    → Blendshapes → Emotion
    → All metrics → Visualizer → Display
    → Every 30 frames → DataUploader → PostgreSQL
    
User Query (Natural Language)
    → MainAgent
        → SQLAgent → PostgreSQL → Session/Log data
        → RAGAgent → Pinecone → Wellness knowledge
    → LLaMA 3.3 → Combined insight
```

---

## 🔷 Knowledge Base Content
4 research documents were vectorized and stored in Pinecone:
| Document | Content Focus |
|---|---|
| `Ergonomics.pdf` | Correct posture, desk setup, ergonomic principles |
| `Fatigue.pdf` | Fatigue causes, measurement, management |
| `Productivity.pdf` | Cognitive performance, breaks, focus optimization |
| `Workplace_enhancement.pdf` | Holistic workplace wellness strategies |

---

## 🔷 Unique Features / Innovations
1. **Personalized Calibration** — Not fixed thresholds; adapts to each user's natural posture
2. **Dual-Detection** — Posture (body) + Fatigue + Emotion simultaneously in one loop
3. **Blendshape Emotion** — No separate ML emotion model; leverages MediaPipe's built-in blendshapes
4. **Parallel AI Agents** — SQL + RAG run concurrently for faster insight generation
5. **Non-intrusive** — Just a webcam; no wearables, no user interruption
6. **Session-based logging** — Historical analysis possible, not just real-time

---

## 🔷 Limitations & Future Scope
**Current Limitations:**
- Single user only (num_poses=1, num_faces=1)
- Emotion detection is rule-based (limited to 3 emotions)
- No mobile/web interface (currently terminal + OpenCV window)
- Requires good lighting for accurate detection

**Future Scope:**
- Multi-user support
- Deep learning–based emotion model (FER datasets)
- Web dashboard with charts for historical analytics
- Mobile app with push notification alerts
- Voice alerts instead of only visual overlays
- Integration with HR systems for team-level wellness reporting

---

# ═══════════════════════════════════════════
# PART 2: REPORT STRUCTURE
# ═══════════════════════════════════════════

```
REPORT TITLE: AI-Powered Real-Time Posture and Emotion Analysis System for Workplace Wellness

ABSTRACT (150–200 words)
  - Problem, approach, methods, key results, conclusion

1. INTRODUCTION
   1.1 Background and Motivation
   1.2 Problem Statement
   1.3 Objectives
   1.4 Scope of the Project
   1.5 Report Organization

2. LITERATURE REVIEW / RELATED WORK
   2.1 Posture Detection Techniques (landmark-based vs. depth camera)
   2.2 Eye Aspect Ratio (EAR) for Drowsiness Detection (Soukupova & Cech, 2016)
   2.3 Facial Emotion Recognition Methods
   2.4 LLM-Based Agents and RAG Systems
   2.5 Summary and Research Gap

3. SYSTEM DESIGN & ARCHITECTURE
   3.1 Overall System Architecture (include diagram)
   3.2 Module Descriptions
       3.2.1 Configuration Module
       3.2.2 Pose Detection Module
       3.2.3 Face Analysis Module (Fatigue + Emotion)
       3.2.4 Visualization Module
       3.2.5 Data Logging Module
       3.2.6 AI Agent Layer
   3.3 Database Schema
   3.4 Data Flow Diagram

4. METHODOLOGY
   4.1 Posture Detection Algorithm
       4.1.1 MediaPipe Pose Landmarker
       4.1.2 Neck Angle Calculation
       4.1.3 Personalized Calibration
   4.2 Fatigue/Drowsiness Detection
       4.2.1 Eye Aspect Ratio (EAR) Algorithm
       4.2.2 Consecutive Frame Threshold Logic
   4.3 Emotion Recognition
       4.3.1 MediaPipe Blendshapes
       4.3.2 Rule-Based Emotion Mapping
   4.4 AI Agent Architecture
       4.4.1 SQL Agent (LangChain + LLaMA 3.3)
       4.4.2 RAG Agent (Pinecone + MiniLM)
       4.4.3 Main Agent (Parallel Orchestration)
   4.5 Knowledge Base Construction

5. IMPLEMENTATION
   5.1 Technology Stack
   5.2 Project Structure
   5.3 Key Code Modules (snippets + explanations)
   5.4 Database Implementation
   5.5 Agent Implementation

6. RESULTS & TESTING
   6.1 System Performance (FPS, detection accuracy)
   6.2 Posture Detection Results
   6.3 Drowsiness Detection Results
   6.4 Emotion Detection Results
   6.5 AI Agent Query Responses (examples)
   6.6 Sample Dashboard Screenshots

7. DISCUSSION
   7.1 Key Findings
   7.2 Challenges Faced
   7.3 Limitations

8. CONCLUSION & FUTURE WORK
   8.1 Conclusion
   8.2 Future Enhancements

REFERENCES
  - MediaPipe (Google)
  - Soukupova & Cech (2016) — EAR Algorithm
  - LangChain Documentation
  - Pinecone Documentation
  - Groq / LLaMA 3 Documentation

APPENDIX
  A. Full Source Code
  B. Database Schema
  C. Sample Outputs / Screenshots
```

---

# ═══════════════════════════════════════════
# PART 3: PPT STRUCTURE (Slide-by-Slide)
# ═══════════════════════════════════════════

```
SLIDE 1 — TITLE SLIDE
  Title: AI-Powered Real-Time Posture & Emotion Analysis System
  Subtitle: Workplace Wellness Through Computer Vision & Generative AI
  Names, Institution, Date

SLIDE 2 — THE PROBLEM
  Visuals: Person slouching at desk, tired eyes
  Points:
  - 80% of desk workers experience back pain
  - Drowsiness reduces productivity by up to 30%
  - Stress/negative emotions go undetected
  - No non-intrusive real-time solution exists

SLIDE 3 — OUR SOLUTION (Overview)
  Visuals: System diagram (simplified)
  - Real-time webcam-based monitoring
  - 3 things detected: Posture | Fatigue | Emotion
  - Logs data to database
  - AI Agent gives personalized wellness advice

SLIDE 4 — SYSTEM ARCHITECTURE
  Visuals: Architecture diagram (draw using your diagram above)
  - Show the flow: Webcam → Detectors → DB → AI Agents → Insight

SLIDE 5 — MODULE 1: POSTURE DETECTION
  Visuals: MediaPipe skeleton visualization, angle diagram
  - MediaPipe Pose: 33 landmarks
  - Neck angle = f(ear, shoulder, hip)
  - Personalized calibration (press 'C')
  - Alert if angle drops 15° below baseline

SLIDE 6 — MODULE 2: FATIGUE DETECTION (EAR)
  Visuals: Eye with EAR formula labeled
  - Eye Aspect Ratio formula
  - Threshold: EAR < 0.15
  - Alert after 20 consecutive frames
  - "DROWSINESS ALERT!" on screen

SLIDE 7 — MODULE 3: EMOTION RECOGNITION
  Visuals: Face with blendshape overlay
  - MediaPipe 52 blendshapes
  - Emotions: Happy | Sad | Surprised | Neutral
  - Rule-based mapping (no separate model needed)
  - Color-coded display on dashboard

SLIDE 8 — REAL-TIME DASHBOARD
  Visuals: Screenshot of the OpenCV window with dashboard
  - Semi-transparent HUD overlay
  - Neck Angle | EAR | Emotion displayed live
  - Alerts in RED
  - FPS counter

SLIDE 9 — DATA LOGGING & DATABASE
  Visuals: Database schema diagram
  - PostgreSQL + SQLAlchemy ORM
  - Two tables: sessions + logdetails
  - Logs every ~1 second (every 30 frames)
  - Stores: angle, EAR, emotion, slouch/drowsy flags

SLIDE 10 — AI AGENT LAYER
  Visuals: Agent flow diagram
  - SQL Agent → queries your session data
  - RAG Agent → searches wellness knowledge base (4 PDFs)
  - Main Agent → runs BOTH in parallel → combined insight
  - LLM: LLaMA 3.3 70B via Groq

SLIDE 11 — KNOWLEDGE BASE (RAG)
  Visuals: PDF icons / Pinecone logo
  - 4 documents vectorized: Ergonomics, Fatigue, Productivity, Workplace Enhancement
  - Stored in Pinecone (cloud vector DB)
  - Embeddings: MiniLM-L6-v2
  - Top-4 chunks retrieved per query

SLIDE 12 — SAMPLE AGENT INTERACTION
  Visuals: Chat-style box
  - Query: "I am slouching too much, how to elucidate it?"
  - SQL Agent: "You slouched 47 times in the last session, avg angle 142°"
  - RAG Agent: "Ergonomic guidelines suggest monitor at eye level..."
  - Main Agent: "Based on your data AND ergonomic research, here's your plan..."

SLIDE 13 — TECH STACK SUMMARY
  Visuals: Logo wall (OpenCV, MediaPipe, LangChain, Pinecone, Groq, PostgreSQL)
  | Layer | Technology |
  | Vision | OpenCV + MediaPipe |
  | Math | NumPy |
  | Database | PostgreSQL + SQLAlchemy |
  | LLM | LLaMA 3.3 70B (Groq) |
  | Agents | LangChain |
  | Vector DB | Pinecone |
  | Embeddings | HuggingFace MiniLM |

SLIDE 14 — RESULTS
  Visuals: Table or chart
  - Real-time processing at ~25-30 FPS
  - Posture: Accurate for 1-2m camera distance
  - Drowsiness: Alert triggers within ~0.7 seconds of eye closure
  - Emotion: 3 core emotions detected with confidence scoring

SLIDE 15 — CHALLENGES & LIMITATIONS
  - Lighting dependency (face/pose detection quality)
  - Single-user only
  - Emotion limited to 3 categories (rule-based)
  - No web/mobile UI (currently desktop only)

SLIDE 16 — FUTURE SCOPE
  - Multi-user support
  - Deep learning emotion model (FER2013)
  - Web dashboard with analytics charts
  - Mobile app with push alerts
  - Voice notifications
  - HR integration for team wellness

SLIDE 17 — CONCLUSION
  - Built a non-intrusive AI wellness monitor
  - Combines CV + LLM + RAG into one pipeline
  - Personalizable, data-driven, actionable insights
  - Ready for extension into enterprise wellness tools

SLIDE 18 — THANK YOU / Q&A
  - Team names
  - GitHub repo (if applicable)
  - Contact info
```

---

# ═══════════════════════════════════════════
# PART 4: QUICK REFERENCE — KEY NUMBERS & FACTS
# ═══════════════════════════════════════════

| Metric | Value |
|---|---|
| Camera Resolution | 1280 × 720 |
| Pose Landmarks | 33 (BlazePose topology) |
| Face Landmarks | 468 |
| Face Blendshapes | 52 |
| EAR Threshold | 0.15 |
| Drowsy Frame Threshold | 20 frames |
| Slouch Margin | 15° below calibrated baseline |
| Default Baseline Angle | 160° |
| Log Interval | Every 30 frames (~1 second) |
| Emotions Detected | Happy, Sad, Surprised, Neutral |
| LLM Model | LLaMA 3.3 70B (via Groq) |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Store | Pinecone (`workplace-knowledge` index) |
| RAG Retrieval (k) | Top 4 chunks |
| Knowledge PDFs | 4 (Ergonomics, Fatigue, Productivity, Workplace Enhancement) |
| Database Tables | 2 (sessions, logdetails) |
| Agents | 3 (SQLAgent, RAGAgent, MainAgent) |
