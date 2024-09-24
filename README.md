Project Overview:

This project aims to develop a face-swapping tool for anonymizing facial features in video streams. By using OpenCV for image processing and MediaPipe FaceMesh for facial landmark detection, the program detects faces, triangulates their features, and seamlessly swaps them for anonymization. This approach addresses privacy concerns, especially in applications like self-driving cars and public surveillance cameras.

Key Components:

  * OpenCV: For video frame capture, image manipulation, and rendering.
  * MediaPipe FaceMesh: To detect facial landmarks, enabling precise feature identification.
  * Triangulation and Warping: Using Delaunay triangulation to map facial features and warp triangles from one face to another.
  * Error Handling: Mechanisms to deal with missing or invalid facial data to ensure the system remains robust.

Main Methods and Workflows:
  
  a. Facial Landmark Detection (get_landmark_points)
  * Detects facial landmarks using MediaPipe and converts the coordinates to pixel values.
  * Handles cases where no face is detected or too many faces are found by exiting or skipping frames.
  
  b. Delaunay Triangulation (get_triangles)
  * Uses Delaunay triangulation to divide facial landmarks into triangles.
  * Helps map source face triangles to destination face triangles for accurate warping.
  
  c. Triangulation and Warping (triangulation, warp_triangle)
  * Each triangle from the source face is extracted, masked, and warped to fit the corresponding triangle in the destination face.
  * Ensures that the warped triangles align with the facial structure of the destination image.
  
  d. Adding Warped Triangles (add_piece_of_new_face)
  * After warping, each triangle is blended seamlessly into the new face.
  * This method ensures consistency in appearance by using masking techniques to apply the triangle precisely.
  
  e. Final Face Swap (swap_new_face)
  * Combines the newly warped face with the destination face using seamless cloning, ensuring a natural and unobtrusive transition.
  * Handles boundary cases like invalid bounding boxes or faces outside image bounds.

Usage:

Install the necessary libraries
``` pip install -r requirements.txt ```

Change the line into your own image at main.py line number 49 

```self.face_swapper = FaceSwapper(src_image_path="images/photo.png")``` 

make sure to add png or jpg extensions and run main.py

![gui](https://github.com/user-attachments/assets/22e744ab-ad8e-48e8-a844-7d7af4223313)
