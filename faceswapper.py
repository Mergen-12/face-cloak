import os
import cv2
import numpy as np
import face_mesh

class FaceSwapper:
    """Class to handle face swapping logic."""
    def __init__(self, src_image_path, width=640, height=480):
        # Constants
        self.WIDTH = width
        self.HEIGHT = height
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, self.WIDTH)
        self.cap.set(4, self.HEIGHT)
        
        # Load source image
        self.src_image = cv2.imread(src_image_path)
        
        # Set initial source image
        self.set_src_image(self.src_image)
    
    def set_src_image(self, image):
        self.src_image = image
        self.src_image_gray = cv2.cvtColor(self.src_image, cv2.COLOR_BGR2GRAY)
        self.src_mask = np.zeros_like(self.src_image_gray)
        
        self.src_landmark_points = face_mesh.get_landmark_points(self.src_image)
        self.src_np_points = np.array(self.src_landmark_points)
        self.src_convexHull = cv2.convexHull(self.src_np_points)
        cv2.fillConvexPoly(self.src_mask, self.src_convexHull, 255)
        
        self.indexes_triangles = face_mesh.get_triangles(
            convexhull=self.src_convexHull,
            landmarks_points=self.src_landmark_points,
            np_points=self.src_np_points
        )

    def process_frame(self, dest_image):
        dest_image_gray = cv2.cvtColor(dest_image, cv2.COLOR_BGR2GRAY)
        dest_mask = np.zeros_like(dest_image_gray)
        
        # Get destination landmark points
        dest_landmark_points = face_mesh.get_landmark_points(dest_image)

        # If no face detected, skip the frame
        if dest_landmark_points is None:
            print("No face detected in the frame, skipping...")
            dest_image_rgb = cv2.cvtColor(dest_image, cv2.COLOR_BGR2RGB)
            return dest_image_rgb
        
        dest_np_points = np.array(dest_landmark_points)
        dest_convexHull = cv2.convexHull(dest_np_points)
        
        height, width, channels = dest_image.shape
        new_face = np.zeros((height, width, channels), np.uint8)
        
        # Triangulation and warping
        for triangle_index in self.indexes_triangles:
            points, src_cropped_triangle, cropped_triangle_mask, _ = face_mesh.triangulation(
                triangle_index=triangle_index,
                landmark_points=self.src_landmark_points,
                img=self.src_image
            )
            
            points2, _, dest_cropped_triangle_mask, rect = face_mesh.triangulation(
                triangle_index=triangle_index,
                landmark_points=dest_landmark_points
            )
            
            warped_triangle = face_mesh.warp_triangle(
                rect=rect, points1=points, points2=points2,
                src_cropped_triangle=src_cropped_triangle,
                dest_cropped_triangle_mask=dest_cropped_triangle_mask
            )
            face_mesh.add_piece_of_new_face(
                new_face=new_face, rect=rect, warped_triangle=warped_triangle
            )
        
        result = face_mesh.swap_new_face(
            dest_image=dest_image, dest_image_gray=dest_image_gray,
            dest_convexHull=dest_convexHull, new_face=new_face
        )
        result = cv2.medianBlur(result, 3)
        
        # Convert BGR to RGB for display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result_rgb
