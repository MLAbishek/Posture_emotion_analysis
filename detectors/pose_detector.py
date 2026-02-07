import mediapipe as mp
import cv2
import numpy as np
import config
from utils.geometry import calculate_angle

class PoseDetector:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.POSE_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        
        # Calibration
        self.neck_baseline = config.SLOUCH_ANGLE_THRESHOLD # Default to config if no calibration
        self.is_calibrated = False

    def set_baseline(self, neck_angle):
        self.neck_baseline = neck_angle
        self.is_calibrated = True
        print(f"Calibration Complete. Baseline Neck Angle: {self.neck_baseline:.1f}")

    def detect(self, image):
        """
        Processes the image and returns pose landmarks.
        Image should be BGR (OpenCV default).
        """
        # Convert to MediaPipe Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        # Note: calling detect() on IMAGE mode. For VIDEO mode, use detect_for_video with timestamp.
        # Since main loop just grabs frames, IMAGE mode is acceptable if we don't need tracking smoothing explicitly reliant on timestamp.
        # However, for smoothing, VIDEO mode is better. But keeping it simple for now.
        detection_result = self.landmarker.detect(mp_image)
        return detection_result

    def calculate_posture_metrics(self, landmarks, image_shape):
        """
        Calculates posture metrics from landmarks.
        landmarks: result.pose_landmarks (list of lists of NormalizedLandmark)
        """
        if not landmarks:
            return None
        
        # landmarks is a list of landmark lists (one per person). We take the first one.
        # Check if empty list
        if len(landmarks) == 0:
            return None
            
        lm = landmarks[0] # List of NormalizedLandmark objects
        h, w, _ = image_shape

        # Helper to get coords
        def get_coords(landmark):
            return [landmark.x * w, landmark.y * h]

        # Key points (indices match Blazepose topology)
        # 11: left_shoulder, 12: right_shoulder
        # 7: left_ear, 8: right_ear
        # 23: left_hip, 24: right_hip
        # 0: nose

        l_shoulder = get_coords(lm[11])
        r_shoulder = get_coords(lm[12])
        l_ear = get_coords(lm[7])
        r_ear = get_coords(lm[8])
        l_hip = get_coords(lm[23])
        nose = get_coords(lm[0])

        # Calculate angles
        # Left side
        neck_angle_l = calculate_angle(l_ear, l_shoulder, l_hip)
        # Right side
        neck_angle_r = calculate_angle(r_ear, r_shoulder, get_coords(lm[24])) # 24: right_hip

        avg_neck_angle = (neck_angle_l + neck_angle_r) / 2
        
        # Calculate deviation from baseline (positive means slouching usually if angle decreases)
        # We assume smaller angle = worse posture (forward head).
        # Trigger if angle is significantly lower than baseline.
        
        return {
            "neck_angle": avg_neck_angle,
            "baseline": self.neck_baseline,
            "is_calibrated": self.is_calibrated,
            "l_shoulder": l_shoulder,
            "r_shoulder": r_shoulder,
            "l_ear": l_ear,
            "r_ear": r_ear,
            "nose": nose
        }
