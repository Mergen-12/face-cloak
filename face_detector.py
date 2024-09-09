import mediapipe as mp
from PIL import Image, ImageTk
import cv2

class FaceDetector:
    """Initializes the FaceDetector with MediaPipe Face Mesh and OpenCV Video Capture."""
    def __init__(self, video_label, static_image_mode=False, max_num_faces=1, device_index=0):
        self.video_label = video_label
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=static_image_mode, max_num_faces=1) #Avoid using too many faces or it will crash

        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open webcam.")
        
        self.frame_number = 0
        self.running = False

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
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    def update_video_label(self, frame):
        """Updates the tkinter label with the OpenCV frame."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        img = Image.fromarray(frame)  # Convert NumPy array to PIL Image
        imgtk = ImageTk.PhotoImage(image=img)  # Convert PIL Image to ImageTk
        self.video_label.imgtk = imgtk  # Keep reference to avoid garbage collection
        self.video_label.configure(image=imgtk)

    def run(self):
        """Runs the face detection process and updates the video feed in the tkinter window."""
        self.running = True
        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            self.frame_number += 1
            if not ret:
                break

            results = self.process_frame(frame)
            self.draw_landmarks(frame, results)

            self.update_video_label(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

        self.cap.release()
        self.face_mesh.close()

    def stop(self):
        """Stops the face detection process."""
        self.running = False