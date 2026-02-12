
import cv2
import dlib
import numpy as np
import sys
from pathlib import Path

def test_image(image_path):
    print(f"Testing image: {image_path}")
    
    # 1. Read Image
    img = cv2.imread(str(image_path))
    if img is None:
        print("Error: Failed to load image (cv2.imread returned None)")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Gray shape: {gray.shape}")
    print(f"Gray dtype: {gray.dtype}")
    print(f"Gray contiguous: {gray.flags['C_CONTIGUOUS']}")
    
    # Force convert to uint8 just in case
    if gray.dtype != np.uint8:
        print("Warning: converting to uint8")
        gray = gray.astype(np.uint8)
        
    # 3. Init dlib
    try:
        detector = dlib.get_frontal_face_detector()
        print("Detector initialized")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return

    # 4. Detect Faces
    try:
        faces = detector(gray, 1)
        print(f"Detection successful. Found {len(faces)} faces.")
    except Exception as e:
        print(f"Error during detection: {e}")
        return

    # 5. Predict Landmarks
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not Path(predictor_path).exists():
        print(f"Predictor not found at {predictor_path}")
        return

    try:
        predictor = dlib.shape_predictor(predictor_path)
        print("Predictor initialized")
        
        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            print("Landmark prediction successful")
        else:
            print("No faces to predict landmarks for")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    target_image = r"d:\github_projects\cg_police2\deepfake_classification\dataset\real\df1_real_id0_0000\frame_0000.jpg"
    test_image(target_image)
