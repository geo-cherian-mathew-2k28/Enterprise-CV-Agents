import cv2
import mediapipe as mp
import numpy as np
import time

class SentinelFlow:
    def __init__(self):
        # 1. Initialize Face Detection (High Speed, High Accuracy)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, # 0 for close range (webcam), 1 for far
            min_detection_confidence=0.6
        )
        
        # 2. System State
        self.security_level = "SECURE" # SECURE, WARNING, BREACH, AWAY
        self.breach_count = 0
        self.last_breach_time = 0
        self.start_time = time.time()
        
        # 3. Visual Config
        self.blur_intensity = (99, 99) # Kernel size for blurring (must be odd)
        self.GREEN = (0, 255, 0)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.CYAN = (255, 255, 0)

    def draw_cyber_hud(self, image, faces, fps):
        h, w, _ = image.shape
        overlay = image.copy()
        
        # --- LOGIC: DETERMINE SECURITY STATE ---
        num_faces = len(faces)
        
        target_color = self.GREEN
        status_text = "SYSTEM SECURE"
        sub_text = "Single User Authenticated"
        
        if num_faces == 0:
            self.security_level = "AWAY"
            target_color = self.YELLOW
            status_text = "USER ABSENT"
            sub_text = "Display Dimmed / Locked"
            # Simulate "Dimming" by darkening image
            image[:] = (image * 0.4).astype(np.uint8)
            
        elif num_faces == 1:
            self.security_level = "SECURE"
            target_color = self.GREEN
            status_text = "SECURE ACCESS"
            sub_text = "Authorized Personnel Only"
            
        elif num_faces > 1:
            self.security_level = "BREACH"
            target_color = self.RED
            status_text = "SECURITY BREACH"
            sub_text = "Multiple Faces Detected - Visual Hacking Attempt"
            self.breach_count += 1
            
            # --- CRITICAL: BLUR SCREEN CONTENT ---
            # We blur the specific region or whole screen to protect data
            # Applying heavy Gaussian Blur to simulate privacy shield
            image[:] = cv2.GaussianBlur(image, self.blur_intensity, 30)

        # --- DRAW VISUALS ---
        
        # 1. Top Bar
        cv2.rectangle(image, (0, 0), (w, 80), (10, 10, 10), -1)
        cv2.line(image, (0, 80), (w, 80), target_color, 2)
        
        # 2. Status Text
        cv2.putText(image, f"STATUS: {status_text}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, target_color, 2)
        
        cv2.putText(image, sub_text, (20, 110), 
                   cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

        # 3. Sidebar Stats
        cv2.rectangle(image, (w-250, 80), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image) # Transparent black
        
        stats = [
            f"FPS: {int(fps)}",
            f"Entities: {num_faces}",
            f"Breaches: {self.breach_count}",
            f"Time: {int(time.time() - self.start_time)}s"
        ]
        
        y_pos = 120
        for stat in stats:
            cv2.putText(image, stat, (w-230, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_pos += 40

        # 4. Face Bounding Boxes (if visible)
        if self.security_level != "BREACH":
            for face in faces:
                bboxC = face.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w_box, h_box = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                     int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw "Targeting" brackets
                color = self.GREEN
                cv2.rectangle(image, (x, y), (x + w_box, y + h_box), color, 2)
                cv2.putText(image, f"{int(face.score[0]*100)}%", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        prev_time = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # 1. Pre-process
            frame = cv2.resize(frame, (1280, 720))
            # Convert to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 2. Inference
            results = self.face_detection.process(image)
            
            # 3. Post-process
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            faces = []
            if results.detections:
                faces = results.detections
            
            # 4. FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # 5. Render Security Interface
            self.draw_cyber_hud(image, faces, fps)
            
            cv2.imshow('SentinelFlow Privacy Shield', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SentinelFlow()
    app.run()