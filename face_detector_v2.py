import cv2
import mediapipe as mp

class FaceDetector:
    """Initializes the FaceDetector with MediaPipe Face Mesh and OpenCV Video Capture."""
    def __init__(self, static_image_mode=False, max_num_faces=1, device_index=0):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=max_num_faces)

        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        
        self.frame_number = 0

    def process_frame(self, image):
        """Processes a single frame and returns the landmarks."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform the face landmarks detection
        results = self.face_mesh.process(image_rgb)
        return results

    def draw_landmarks(self, image, results):
        """Draws the face landmarks on the image."""
        if results.multi_face_landmarks:
            h, w, _ = image.shape
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    print(f"Frame Number {self.frame_number} | Landmark x:{x}, y:{y}")
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    def run(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                self.frame_number += 1
                if not ret:
                    break

                results = self.process_frame(frame)
                
                self.draw_landmarks(frame, results)
                cv2.putText(frame, f"Frame: {self.frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Landmarks', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            self.face_mesh.close()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    face_detector = FaceDetector()
    face_detector.run()
