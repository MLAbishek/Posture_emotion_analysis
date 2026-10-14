# Module-Level Technical Documentation

This document provides a detailed, module-by-module breakdown of the AI-Powered Posture and Emotion Analysis System. It is intended for developers and maintainers to understand the internal workings, classes, methods, and configurations of the project.

## Directory Structure

```text
backend/
├── main.py
├── config.py
├── db_config.py
├── Agents/
│   ├── MainAgent.py
│   ├── RAGAgent.py
│   └── SQLAgent.py
├── detectors/
│   ├── face_analyzer.py
│   └── pose_detector.py
├── knowledge_base_making/
│   └── chunker.py
└── utils/
    ├── DataUploader.py
    ├── geometry.py
    └── visualizer.py
```

## 1. Core Application

### `main.py`
The entry point of the application. It runs the primary loop that continuously captures frames from the webcam, processes them through the various detectors, updates the visualizer, and logs the metrics to the database.

- **Initialization:** Sets up the camera, `PoseDetector`, `FaceAnalyzer`, `Visualizer`, and `DataUploader`. Starts a new database session.
- **Calibration Phase (`calibration_mode`):** Waits for the user to press 'c'. When pressed, it calls `pose_detector.set_baseline()` using the current neck angle.
- **Main Loop:**
  - Reads camera frame.
  - Calls `pose_detector.detect()` and `pose_detector.calculate_posture_metrics()`.
  - Calls `face_analyzer.detect()` and `face_analyzer.analyze()`.
  - Processes drowsiness logic using EAR (Eye Aspect Ratio). If EAR is below `config.EAR_THRESHOLD` for `config.EAR_CONSECUTIVE_FRAMES`, it flags `is_drowsy`.
  - Logs aggregated metrics using `data_uploader.log_data()`.
  - Displays the UI overlay via `visualizer.draw_dashboard()`.

### `config.py`
Centralized configuration file defining constants and thresholds used across the system.

- **Thresholds:**
  - `EAR_THRESHOLD`: Eye Aspect Ratio threshold for drowsiness (0.15).
  - `EAR_CONSECUTIVE_FRAMES`: Frames required below threshold to trigger alert (20).
  - `SLOUCH_ANGLE_THRESHOLD`: Default neck angle (160).
  - `SLOUCH_THRESHOLD_MARGIN`: Deviation from baseline before triggering an alert (15).
- **Camera Settings:** Defines `CAMERA_ID`, `FRAME_WIDTH` (1280), and `FRAME_HEIGHT` (720).
- **Model Paths:** Resolves absolute paths to MediaPipe models (`pose_landmarker_full.task` and `face_landmarker.task`).
- **UI Styling:** Defines RGB tuples for colors and OpenCV font constants.

### `db_config.py`
Configures the connection to the PostgreSQL database using SQLAlchemy.

- Loads environment variables (`.env`) for `DATABASE_URL`.
- Initializes `create_engine` and `sessionmaker` (`SessionLocal`).
- **`get_db()`:** A generator function providing a transactional scope around a session.

---

## 2. Detectors (`detectors/`)

### `detectors/pose_detector.py`
Responsible for skeletal detection and posture metrics calculation.

- **Class `PoseDetector`**
  - **`__init__()`:** Initializes the MediaPipe Pose Landmarker with the pre-trained `.task` model. Sets default `neck_baseline`.
  - **`set_baseline(neck_angle)`:** Calibrates the user's baseline posture angle.
  - **`detect(image)`:** Converts the OpenCV BGR image to RGB and processes it through the MediaPipe landmarker.
  - **`calculate_posture_metrics(landmarks, image_shape)`:**
    - Extracts key coordinates: shoulders, ears, hips, nose.
    - Calculates the neck angle on both the left and right sides using `geometry.calculate_angle()`.
    - Returns a dictionary with `neck_angle` (average), `baseline`, calibration status, and raw landmark coordinates.

### `detectors/face_analyzer.py`
Responsible for analyzing facial landmarks for both fatigue (drowsiness) and emotions.

- **Class `FaceAnalyzer`**
  - **`__init__()`:** Initializes MediaPipe Face Landmarker with `output_face_blendshapes=True`. Defines landmark index arrays for the Left and Right eyes.
  - **`detect(image)`:** Runs the face landmarker on the given image.
  - **`analyze(result, image_shape)`:** Aggregates and returns both EAR metrics and emotion metrics.
  - **`calculate_ear(landmarks, image_shape)`:**
    - Computes Eye Aspect Ratio (EAR) for both eyes using `geometry.euclidean_distance()`.
    - Formula: `(vertical_1 + vertical_2) / (2.0 * horizontal)`.
  - **`detect_emotion(blendshapes)`:**
    - Parses MediaPipe blendshape categories (52 total action units).
    - **Happy**: Combines left and right mouth smiles.
    - **Sad**: Combines mouth frowns and inner brow raises.
    - **Surprise**: Combines outer brow raises and jaw openness (mandates `jawOpen > 0.25`).
    - Defaults to "Neutral" if the dominant emotion confidence is below 0.2.

---

## 3. Utilities (`utils/`)

### `utils/DataUploader.py`
Manages the SQLAlchemy Object-Relational Mapping (ORM) and data ingestion into PostgreSQL.

- **Schema Definitions (`Base`)**
  - `SessionModel` (`sessions`): Tracks `id`, `start_time`, `end_time`.
  - `LogModel` (`logdetails`): Tracks metrics per frame tick (neck angle, slouch flag, EAR, drowsy flag, emotion, and emotion confidence) linked via `session_id`.
- **Class `DataUploader`**
  - **`start_session()`:** Creates a new `SessionModel` record.
  - **`log_data(...)`:** Logs metrics to `LogModel`. Uses a `frame_counter` to only push data every 30 frames (controlled by `self.log_interval`), preventing database overload.
  - **`end_session()`:** Updates the `SessionModel` with the current `end_time`.

### `utils/visualizer.py`
Handles all OpenCV drawing operations to create the Heads-Up Display (HUD).

- **Class `Visualizer`**
  - **`draw_dashboard(image, posture_data, fatigue_data, emotion_data, calibration_mode, is_drowsy)`:**
    - Draws a semi-transparent black overlay at the top of the frame.
    - If `calibration_mode` is True, it displays calibration instructions.
    - Dynamically displays text metrics (Neck Angle, EAR, Emotion).
    - Changes text color to RED and triggers alerts ("POOR POSTURE!", "DROWSINESS ALERT!") when thresholds are breached.
    - Draws a line connecting the user's ear to their shoulder to visualize the posture angle.

### `utils/geometry.py`
Provides foundational math utilities.

- **`calculate_angle(a, b, c)`:** Uses `np.arctan2` to calculate the internal angle (in degrees) given three points, where `b` is the vertex.
- **`euclidean_distance(p1, p2)`:** Computes standard straight-line distance between two 2D points.

---

## 4. Agents (`Agents/`)

This directory contains the AI Agent layer responsible for answering user queries by combining historical SQL analytics with knowledge base retrieval.

### `Agents/SQLAgent.py`
Creates a LangChain SQL Agent that queries the PostgreSQL database.

- **LLM:** `llama-3.3-70b-versatile` via Groq.
- Connects to the database using `SQLDatabase.from_uri`.
- Uses a custom system prompt ensuring it writes syntactically correct queries, limits output (`top_k=5`), and crucially, **DOES NOT execute DML statements** (read-only constraint).
- **`sqlagent(query)`:** Takes a natural language string and returns analytical answers from the `logdetails` and `sessions` tables.

### `Agents/RAGAgent.py`
Retrieval-Augmented Generation (RAG) agent for referencing workplace wellness literature.

- Connects to a **Pinecone** vector database (`workplace-knowledge` index).
- Uses `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace for embedding queries.
- Retrieves the top 4 (`k=4`) most relevant document chunks based on cosine similarity.
- **`ragagent(query)`:** Injects retrieved context into a prompt for the LLaMA 3.3 model to answer the user's question accurately.

### `Agents/MainAgent.py`
The orchestrator that runs the SQL and RAG agents concurrently.

- Uses LangChain's `RunnableParallel` to execute both `sql_runnable` and `rag_runnable` at the same time.
- **`mainagent(query)`:**
  1. Invokes the parallel runnable.
  2. Receives the data-driven answer from `SQLAgent`.
  3. Receives the context-driven answer from `RAGAgent`.
  4. Feeds both outputs into a final synthesized prompt for the LLaMA 3.3 model to combine the statistical reality (SQL) with expert advice (RAG).

---

## 5. Knowledge Base Construction (`knowledge_base_making/`)

### `knowledge_base_making/chunker.py`
A utility script used to ingest PDF documents into the Pinecone Vector Database.

- **Document Loading:** Uses `PyPDFLoader` to read local wellness PDFs (Ergonomics, Fatigue, Productivity, Workplace Enhancement).
- **Chunking:** Uses `RecursiveCharacterTextSplitter` to break down documents into 1000-character chunks with a 200-character overlap.
- **Metadata Tagging:** Assigns custom metadata (`source`, `topic`) to chunks for better retrieval filtering later.
- **Indexing:** Uses `PineconeVectorStore` and HuggingFace embeddings to push the chunked vectors to the AWS-hosted Pinecone index (`workplace-knowledge`).
