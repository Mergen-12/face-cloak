import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize OpenCV video capture.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Convert the image to RGB as MediaPipe uses RGB images.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the face landmarks detection.
    results = face_mesh.process(image_rgb)
    
    # Draw the face landmarks.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Get the position of the landmark.
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                
                # Draw the landmark on the image.
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                
                # Optionally, you can label the landmarks with their index numbers.
                cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # Display the output.
    cv2.imshow('Face Landmarks', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
