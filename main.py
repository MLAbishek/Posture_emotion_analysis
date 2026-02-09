import cv2
import time
import config
from detectors.pose_detector import PoseDetector
from detectors.face_analyzer import FaceAnalyzer
from utils.visualizer import Visualizer
from utils.DataUploader import DataUploader

def main():
    # Initialize Camera
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    pose_detector = PoseDetector()
    face_analyzer = FaceAnalyzer()
    visualizer = Visualizer()
    data_uploader = DataUploader()
    data_uploader.start_session()
    
    prev_time = 0

    print("Starting Posture & Fatigue Detection... Press 'q' to quit.")
    print("Press 'c' to calibrate posture.")


    is_calibrated = False
    calibration_mode = True
    
    
    drowsy_frame_count = 0
    is_drowsy = False

    while True:
        try:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Performance measurement
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            image_shape = image.shape

            #Pose Detection
            pose_results = pose_detector.detect(image)
            posture_metrics = None
            if pose_results and pose_results.pose_landmarks:
                posture_metrics = pose_detector.calculate_posture_metrics(pose_results.pose_landmarks, image_shape)

            # CALIBRATION LOGIC
            if calibration_mode:
                
                
                key = cv2.waitKey(5) & 0xFF
                if key == ord('c'):
                    if posture_metrics:
                        pose_detector.set_baseline(posture_metrics['neck_angle'])
                        calibration_mode = False
                        is_calibrated = True
                        print("Calibration done.")
                    else:
                        print("No pose detected. Cannot calibrate.")

                image = visualizer.draw_dashboard(image, posture_metrics, None, None, calibration_mode=True)
                
                cv2.imshow('Posture & Fatigue Analysis', image)
                
                if key == ord('q'):
                    break
                continue 

            #Face Analysis (Fatigue & Emotion)
            face_results = face_analyzer.detect(image)
            face_metrics = None
            if face_results:
                face_metrics = face_analyzer.analyze(face_results, image_shape)

            #Visualization
            # Unwrap metrics if they exist
            fatigue_data = face_metrics['fatigue_data'] if face_metrics else None
            emotion_data = face_metrics['emotion_data'] if face_metrics else None

            #Drowsiness Logic 
            if fatigue_data:
                ear = fatigue_data['avg_ear']
                
                if ear < config.EAR_THRESHOLD:
                    drowsy_frame_count += 1
                else:
                    drowsy_frame_count = 0
                
                if drowsy_frame_count >= config.EAR_CONSECUTIVE_FRAMES:
                    is_drowsy = True
                else:
                    is_drowsy = False
            else:
                drowsy_frame_count = 0
                is_drowsy = False

            # Log Data to DB (After all metrics are calculated)
            if posture_metrics and fatigue_data and emotion_data:
                # Calculate slouching based on dynamic baseline
                baseline = posture_metrics.get('baseline', config.SLOUCH_ANGLE_THRESHOLD)
                threshold = baseline - config.SLOUCH_THRESHOLD_MARGIN
                is_slouching = posture_metrics['neck_angle'] < threshold

                data_uploader.log_data(
                    neck_angle=posture_metrics['neck_angle'],
                    is_slouching=is_slouching,
                    ear_value=fatigue_data['avg_ear'],
                    is_drowsy=is_drowsy,
                    emotion=emotion_data['dominant_emotion'],
                    emotion_confidence=emotion_data['confidence']
                )

            image = visualizer.draw_dashboard(image, posture_metrics, fatigue_data, emotion_data, calibration_mode=False, is_drowsy=is_drowsy)


            cv2.putText(image, f"FPS: {int(fps)}", (20, 700), config.FONT, 0.7, config.GREEN, 2)

            cv2.imshow('Posture & Fatigue Analysis', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    data_uploader.end_session()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
