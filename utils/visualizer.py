import cv2
import numpy as np
import config

class Visualizer:
    def __init__(self):
        pass

    def draw_dashboard(self, image, posture_data, fatigue_data, emotion_data=None, calibration_mode=False, is_drowsy=False):
        """
        Draws metrics and alerts on the image.
        """
        h, w, _ = image.shape
        
        # Create a dashboard background (semi-transparent)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        camera_alpha = 0.6
        cv2.addWeighted(overlay, camera_alpha, image, 1 - camera_alpha, 0, image)

        # Calibration UI
        if calibration_mode:
            cv2.putText(image, "CALIBRATION MODE", (w//2 - 150, 40), config.FONT, 1.0, config.YELLOW, 2)
            cv2.putText(image, "Sit Straight & Press 'C' to Calibrate", (w//2 - 250, 80), config.FONT, 0.8, config.WHITE, 2)
            
            if posture_data:
                 cv2.putText(image, f"Current Angle: {int(posture_data['neck_angle'])}", (w//2 - 100, 120), config.FONT, 0.7, config.GREEN, 2)
            return image

        # Draw metrics
        x_start = 20
        y_start = 40
        
        # Posture Info
        neck_angle = 0
        if posture_data:
            neck_angle = posture_data['neck_angle']
            baseline = posture_data.get('baseline', config.SLOUCH_ANGLE_THRESHOLD)
            is_calibrated = posture_data.get('is_calibrated', False)
            
            # Dynamic threshold
            threshold = baseline - config.SLOUCH_THRESHOLD_MARGIN
            
            color = config.GREEN
            if neck_angle < threshold:
                color = config.RED
                cv2.putText(image, "POOR POSTURE!", (w - 300, 50), config.FONT, 1, config.RED, 2)
            
            text = f"Neck Angle: {int(neck_angle)} deg"
            if is_calibrated:
                text += f" (Base: {int(baseline)})"
            
            cv2.putText(image, text, (x_start, y_start), config.FONT, 0.7, color, 2)
            
            if 'l_shoulder' in posture_data:
                p1 = tuple(map(int, posture_data['l_ear']))
                p2 = tuple(map(int, posture_data['l_shoulder']))
                cv2.line(image, p1, p2, color, 2)

        # Fatigue Info
        ear = 0
        if fatigue_data:
            ear = fatigue_data['avg_ear']
            color = config.GREEN
            if is_drowsy:
                color = config.RED
                cv2.putText(image, "DROWSINESS ALERT!", (w - 350, 90), config.FONT, 1, config.RED, 2)
            
            # Move EAR to the next line to avoid overlap with Posture info
            text = f"EAR: {ear:.2f}"
            cv2.putText(image, text, (x_start, y_start + 40), config.FONT, 0.7, color, 2)

        # Emotion Info
        if emotion_data:
            emotion = emotion_data['dominant_emotion']
            confidence = emotion_data['confidence']
            
            # Color coding for emotions
            e_color = config.YELLOW
            if emotion == "Happy":
                e_color = config.GREEN
            elif emotion == "Sad":
                e_color = config.BLUE
            elif emotion == "Surprise":
                e_color = config.WHITE

            text = f"Emotion: {emotion} ({confidence:.2f})"
            # Move Emotion to the line below EAR
            cv2.putText(image, text, (x_start, y_start + 80), config.FONT, 0.7, e_color, 2)

        return image
