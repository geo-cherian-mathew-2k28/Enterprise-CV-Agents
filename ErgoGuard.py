import cv2
import mediapipe as mp
import numpy as np
import time
import math

class ErgoGuard:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Application State
        self.good_posture_time = 0
        self.bad_posture_time = 0
        self.start_time = time.time()
        self.status = "Good"
        self.frame_count = 0
        
        # Colors (BGR)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.BLUE = (255, 0, 0)

    def calculate_angle(self, a, b, c):
        """Calculates the angle between three points (shoulder, ear, hip for example)"""
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def draw_hud(self, image, score, neck_inclination, torso_inclination):
        """Draws a futuristic Interface"""
        h, w, _ = image.shape
        
        # 1. Status Bar at Top
        color = self.GREEN if self.status == "Good" else self.RED
        cv2.rectangle(image, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(image, f"STATUS: {self.status.upper()}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 2. Metrics Panel (Left Side)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 60), (250, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Data Points
        y_pos = 100
        stats = [
            f"Neck Angle: {int(neck_inclination)} deg",
            f"Torso Angle: {int(torso_inclination)} deg",
            f"FPS: {int(self.fps)}",
            f"Focus Score: {int(score)}%"
        ]
        
        for stat in stats:
            cv2.putText(image, stat, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 40
            
        # 3. Dynamic Health Bar
        cv2.rectangle(image, (15, y_pos), (235, y_pos+20), (50, 50, 50), -1)
        bar_width = int((score / 100) * 220)
        bar_color = self.GREEN if score > 70 else (self.YELLOW if score > 40 else self.RED)
        cv2.rectangle(image, (15, y_pos), (15 + bar_width, y_pos+20), bar_color, -1)
        cv2.putText(image, "Health Integrity", (15, y_pos-10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

    def process_frame(self, frame):
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = self.pose.process(image)
        
        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        h, w, _ = image.shape
        
        neck_inclination = 0
        torso_inclination = 0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            
            l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * w,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * h]
            
            l_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                     landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            
            # --- LOGIC 1: NECK INCLINATION ---
            # Angle between Shoulder vertical axis and Ear
            # We create a virtual point directly above the shoulder to calculate inclination
            # Checking Left Side only for simplicity (assume symmetry)
            neck_inclination = self.calculate_angle(l_ear, l_shoulder, [l_shoulder[0], l_shoulder[1] - 100])
            
            # --- LOGIC 2: TORSO INCLINATION ---
            # Angle between Hip vertical axis and Shoulder
            torso_inclination = self.calculate_angle(l_shoulder, l_hip, [l_hip[0], l_hip[1] - 100])
            
            # --- HEURISTICS FOR "BAD POSTURE" ---
            # Neck > 40 degrees usually means looking down too much
            # Torso > 10 degrees means leaning back/forward too much
            if neck_inclination > 40 or torso_inclination > 15:
                self.status = "Bad Posture"
                self.bad_posture_time += 1
                color = self.RED
            else:
                self.status = "Good"
                self.good_posture_time += 1
                color = self.GREEN
                
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )
            
            # Visualize angles
            cv2.putText(image, str(int(neck_inclination)), tuple(np.multiply(l_shoulder, [1, 1]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Calculate Score
        total_time = self.good_posture_time + self.bad_posture_time
        score = 100
        if total_time > 0:
            score = (self.good_posture_time / total_time) * 100
            
        return image, score, neck_inclination, torso_inclination

    def run(self):
        cap = cv2.VideoCapture(0)
        
        # Calculate FPS
        prev_frame_time = 0
        new_frame_time = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize for consistent processing
            frame = cv2.resize(frame, (1280, 720))
            
            # Process
            image, score, neck, torso = self.process_frame(frame)
            
            # FPS Calculation
            new_frame_time = time.time()
            self.fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            # Draw HUD
            self.draw_hud(image, score, neck, torso)
            
            cv2.imshow('ErgoGuard AI', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ErgoGuard()
    app.run()
