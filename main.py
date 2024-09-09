import cv2
import mediapipe as mp
import threading
import tkinter as tk
from tkinter import messagebox
from face_detector import FaceDetector
import time


class FaceDetectorGUI:
    """FaceDetectorGUI handles the GUI for the FaceDetector."""
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detector GUI")
        
        # Create Start and Stop buttons
        self.start_button = tk.Button(root, text="Start Detection", command=self.start_detection, width=25, height=2)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Detection", command=self.stop_detection, width=25, height=2)
        self.stop_button.pack(pady=10)

        # Video label to display OpenCV feed inside tkinter window
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.detector = None
        self.thread = None

    def start_detection(self):
        """Starts face detection in a separate thread."""
        if not self.detector:
            self.detector = FaceDetector(self.video_label)
        
        self.thread = threading.Thread(target=self.detector.run)
        self.thread.daemon = True
        self.thread.start()

    def stop_detection(self):
        """Stops face detection."""
        if self.detector:
            self.detector.stop()
            self.is_running = False

            # Poll the thread until it finishes properly, ensuring resources are released
            while self.thread.is_alive():
                time.sleep(0.1)  # Allow the thread to finish properly cuz threads
            
            # Clear the video label after stopping
            self.video_label.config(image='')
            messagebox.showinfo("Info", "Face detection stopped.")

        self.detector = None  # Reset detector object

if __name__ == "__main__":
    root = tk.Tk()
    gui = FaceDetectorGUI(root)
    root.mainloop()