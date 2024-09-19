import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import cv2
import mediapipe as mp
import numpy as np

def get_landmark_points(src_image):
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1, # might wanna increase here
            refine_landmarks=True) as face_mesh:
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return None
        if len(results.multi_face_landmarks) > 1:
            sys.exit("There are too much face landmarks")

        src_face_landmark = results.multi_face_landmarks[0].landmark
        landmark_points = []
        for i in range(468):
            y = int(src_face_landmark[i].y * src_image.shape[0])
            x = int(src_face_landmark[i].x * src_image.shape[1])
            landmark_points.append((x, y))
        return landmark_points


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def get_triangles(convexhull, landmarks_points, np_points):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((np_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)
        index_pt2 = np.where((np_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)
        index_pt3 = np.where((np_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    return indexes_triangles


def triangulation(triangle_index, landmark_points, img=None):
    tr1_pt1 = landmark_points[triangle_index[0]]
    tr1_pt2 = landmark_points[triangle_index[1]]
    tr1_pt3 = landmark_points[triangle_index[2]]
    triangle = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect = cv2.boundingRect(triangle)
    (x, y, w, h) = rect

    cropped_triangle = None
    if img is not None:
        cropped_triangle = img[y: y + h, x: x + w]

    cropped_triangle_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

    return points, cropped_triangle, cropped_triangle_mask, rect


def warp_triangle(rect, points1, points2, src_cropped_triangle, dest_cropped_triangle_mask):
    (x, y, w, h) = rect
    matrix = cv2.getAffineTransform(np.float32(points1), np.float32(points2))
    warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=dest_cropped_triangle_mask)
    return warped_triangle


def add_piece_of_new_face(new_face, rect, warped_triangle):
    (x, y, w, h) = rect
    
    if w == 0 or h == 0:
        if not hasattr(add_piece_of_new_face, 'warning_shown'):
            print("Warning: Invalid face region dimensions.")
            add_piece_of_new_face.warning_shown = True
        return
    
    new_face_rect_area = new_face[y: y + h, x: x + w]
    
    if new_face_rect_area is None or new_face_rect_area.size == 0:
        if not hasattr(add_piece_of_new_face, 'warning_shown'):
            print("Warning: Empty face region.")
            add_piece_of_new_face.warning_shown = True
        return

    # Convert to float32 for precision if necessary
    new_face_rect_area = new_face_rect_area.astype('float32')
    warped_triangle = warped_triangle.astype('float32')

    new_face_rect_area_gray = cv2.cvtColor(new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)

    # Ensure mask and triangle have the same size
    if warped_triangle.shape[:2] != mask_triangles_designed.shape[:2]:
        mask_triangles_designed = cv2.resize(mask_triangles_designed, (warped_triangle.shape[1], warped_triangle.shape[0]))

    # Apply the mask
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed.astype('uint8'))

    # Ensure both have the same size
    if warped_triangle.shape[:2] != new_face_rect_area.shape[:2]:
        warped_triangle = cv2.resize(warped_triangle, (new_face_rect_area.shape[1], new_face_rect_area.shape[0]))

    # Ensure both have the same number of channels
    if len(warped_triangle.shape) == 2:  # If warped_triangle is grayscale
        warped_triangle = cv2.cvtColor(warped_triangle, cv2.COLOR_GRAY2BGR)

    # Add the warped triangle to the face region
    new_face_rect_area = cv2.add(new_face_rect_area, warped_triangle)

    # Convert back to uint8 if needed
    new_face[y: y + h, x: x + w] = cv2.convertScaleAbs(new_face_rect_area)

def swap_new_face(dest_image, dest_image_gray, dest_convexHull, new_face):
    face_mask = np.zeros_like(dest_image_gray)
    head_mask = cv2.fillConvexPoly(face_mask, dest_convexHull, 255)
    face_mask = cv2.bitwise_not(head_mask)

    head_without_face = cv2.bitwise_and(dest_image, dest_image, mask=face_mask)
    result = cv2.add(head_without_face, new_face)

    # Calculate bounding rect for destination convex hull
    (x, y, w, h) = cv2.boundingRect(dest_convexHull)
    
    # Ensure bounding box fits within the image dimensions
    img_height, img_width = dest_image.shape[:2]
    
    # Ensure the coordinates are within image boundaries
    if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
        print(f"Warning: Invalid bounding box (x, y, w, h): ({x}, {y}, {w}, {h})")
        return dest_image  # Return the original image without any changes

    center_face = (int((x + x + w) / 2), int((y + y + h) / 2))

    # Ensure center face is within bounds
    if center_face[0] < 0 or center_face[1] < 0 or center_face[0] >= img_width or center_face[1] >= img_height:
        print(f"Warning: Invalid center face position: {center_face}")
        return dest_image

    # Seamless clone only if everything is valid
    try:
        return cv2.seamlessClone(result, dest_image, head_mask, center_face, cv2.NORMAL_CLONE)
    except cv2.error as e:
        print(f"Error during seamless cloning: {str(e)}")
        return dest_image
