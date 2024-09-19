import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from faceswapper import FaceSwapper
from log import ConsoleLogHandler

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set window title and size
        self.setWindowTitle("Face Swapper")
        self.setGeometry(100, 100, 800, 800)

        # Main layout
        self.layout = QVBoxLayout()

        # Start and Stop buttons
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_swapping)
        self.layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_swapping)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        # Video feed (face region)
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: white;")
        self.layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Console log widget
        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setFixedHeight(150)
        self.layout.addWidget(self.console_log)

        # Redirect stdout and stderr to console_log
        sys.stdout = ConsoleLogHandler(self.console_log)
        sys.stderr = ConsoleLogHandler(self.console_log)

        # Set layout to the main window
        self.setLayout(self.layout)

        # Face Swapper
        self.face_swapper = FaceSwapper(src_image_path="images/01047.png")

        # Video capture initialization
        self.cap = self.face_swapper.cap
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def start_swapping(self):
        """Start swapping and enable video capture."""
        self.timer.start(30)  # 30ms per frame update
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_message("Swapping started.")

    def stop_swapping(self):
        """Stop swapping and release resources."""
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Swapping stopped.")

    def update_frame(self):
        """Capture video frame, process it, and update the video feed."""
        ret, frame = self.cap.read()
        if not ret:
            self.log_message("Failed to capture frame.")
            self.stop_swapping()
            return

        # Process the frame using the FaceSwapper class
        result_frame = self.face_swapper.process_frame(frame)

        # Convert the frame to QImage for displaying in the QLabel
        height, width, channel = result_frame.shape
        bytes_per_line = channel * width
        q_img = QImage(result_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        # Set the image to the QLabel (video feed)
        self.video_label.setPixmap(pixmap)

    def log_message(self, message):
        """Helper function to add messages to the console log (simulating terminal)."""
        self.console_log.append(message)
        self.console_log.moveCursor(self.console_log.textCursor().End)  # Auto-scroll to the end

    def closeEvent(self, event):
        """Handle the close event to clean up resources."""
        self.stop_swapping()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())