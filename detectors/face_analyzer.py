import mediapipe as mp
import cv2
import numpy as np
import config
from utils.geometry import euclidean_distance

class FaceAnalyzer:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Enable blendshapes for emotion detection
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.FACE_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        
        # Landmark indices for Left Eye
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144] 
        # Landmark indices for Right Eye
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def detect(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        detection_result = self.landmarker.detect(mp_image)
        return detection_result

    def analyze(self, result, image_shape):
        """
        Analyzes the detection result for both Fatigue and Emotion.
        Returns a dictionary with 'ear_metrics' and 'emotion_metrics'.
        """
        if not result or not result.face_landmarks:
            return None

        # 1. Calculate Fatigue (EAR)
        ear_metrics = self.calculate_ear(result.face_landmarks, image_shape)
        
        # 2. Calculate Emotion (Blendshapes)
        emotion_metrics = self.detect_emotion(result.face_blendshapes)
        
        return {
            "fatigue_data": ear_metrics,
            "emotion_data": emotion_metrics
        }

    def detect_emotion(self, blendshapes):
        """
        Interprets blendshapes to detect basic emotions.
        blendshapes: list of FaceBlendshapesCategories (usually one list per face)
        """
        if not blendshapes:
            return None
        
        # Get the first face's blendshapes
        bs_list = blendshapes[0] # This is a list of categories
        
        # Convert to a dictionary for easier access {category_name: score}
        scores = {b.category_name: b.score for b in bs_list}
        
        # Simple Logic for basic emotions
        # 1. Happiness: mouthSmileLeft + mouthSmileRight
        happiness = (scores.get('mouthSmileLeft', 0) + scores.get('mouthSmileRight', 0)) / 2.0
        
        # 2. Sadness: mouthFrownLeft + mouthFrownRight
        sadness = (scores.get('mouthFrownLeft', 0) + scores.get('mouthFrownRight', 0)) / 2.0
        
        # 3. Surprise: browInnerUp + jawOpen
        surprise = (scores.get('browInnerUp', 0) + scores.get('jawOpen', 0)) / 2.0
        
        # 4. Neutral (if everything is low) or specific logic.
        
        emotions = {
            "Happy": happiness,
            "Sad": sadness,
            "Surprise": surprise
        }
        
        # Find dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        confidence = emotions[dominant_emotion]
        
        if confidence < 0.1: # Threshold for "Neutral"
            dominant_emotion = "Neutral"
            
        return {
            "dominant_emotion": dominant_emotion,
            "confidence": confidence,
            "scores": emotions
        }

    def calculate_ear(self, landmarks, image_shape):
        """
        Calculates Eye Aspect Ratio (EAR) for both eyes.
        landmarks: result.face_landmarks (list of lists of NormalizedLandmark)
        """
        if not landmarks or len(landmarks) == 0:
            return None

        lm = landmarks[0] # First face
        h, w, _ = image_shape

        def get_coords(index):
            landmark = lm[index]
            return np.array([landmark.x * w, landmark.y * h])

        def eye_aspect_ratio(indices):
            # indices: p1, p2, p3, p4, p5, p6
            p1 = get_coords(indices[0])
            p2 = get_coords(indices[1])
            p3 = get_coords(indices[2])
            p4 = get_coords(indices[3])
            p5 = get_coords(indices[4])
            p6 = get_coords(indices[5])

            # Calculate distances
            vertical_1 = euclidean_distance(p2, p6)
            vertical_2 = euclidean_distance(p3, p5)
            horizontal = euclidean_distance(p1, p4)

            # Avoid division by zero
            if horizontal == 0:
                return 0.0

            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear

        left_ear = eye_aspect_ratio(self.LEFT_EYE)
        right_ear = eye_aspect_ratio(self.RIGHT_EYE)
        
        avg_ear = (left_ear + right_ear) / 2.0
        
        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear
        }
