import cv2

# Thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for drowsiness
EAR_CONSECUTIVE_FRAMES = 20  # Number of frames below threshold to trigger alert

SLOUCH_ANGLE_THRESHOLD = 160  # Angle threshold for neck inclination (degrees)
SLOUCH_THRESHOLD_MARGIN = 15 # Degrees below baseline to trigger alert
NECK_INCLINATION_THRESHOLD = 40 # Threshold for forward head posture

# Camera
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Models
POSE_MODEL_PATH = r"d:\Posture_emotion_analysis\models\pose_landmarker_full.task"
FACE_MODEL_PATH = r"d:\Posture_emotion_analysis\models\face_landmarker.task"

# Colors (B, G, R)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

# Fonts
FONT = cv2.FONT_HERSHEY_SIMPLEX
