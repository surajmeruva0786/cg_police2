import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import os
import pandas as pd
import glob
from tqdm import tqdm

def extract_features_from_image(image_path):
    """
    Extract all 30 features from a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to the face image
    
    Returns:
    --------
    dict
        Dictionary containing all 30 features
    """
    # Load the image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create mask from non-black pixels
    mask = np.any(img > 0, axis=2).astype(np.uint8) * 255
    
    # Initialize features dictionary
    features = {}
    
    # Extract facial landmark features
    _, landmark_features = extract_landmark_features(img, mask)
    features['Eye_Ratio'] = landmark_features[0]
    features['Height_Width_Ratio'] = landmark_features[1]
    features['Eye_Mouth_Ratio'] = landmark_features[2]
    features['Eye_Distance'] = landmark_features[3]
    features['Face_Width'] = landmark_features[4]
    features['Face_Height'] = landmark_features[5]
    features['Eye_to_Mouth_Distance'] = landmark_features[6]
    
    # Extract illumination features
    _, illum_features = extract_illumination_features(img, mask)
    features['Primary_Light_Direction'] = illum_features[0]
    features['Direction_Consistency'] = illum_features[1]
    features['Highlight_Shadow_Ratio'] = illum_features[2]
    features['Local_Contrast'] = illum_features[3]
    features['Illumination_Uniformity'] = illum_features[4]
    features['Average_Gradient_Magnitude'] = illum_features[5]
    
    # Extract color features
    _, color_features = extract_color_features(img, mask)
    features['Hue_Mean'] = color_features[0]
    features['Hue_Std'] = color_features[1]
    features['Saturation_Mean'] = color_features[2]
    features['Saturation_Std'] = color_features[3]
    features['Value_Mean'] = color_features[4]
    features['Value_Std'] = color_features[5]
    features['Color_Contrast'] = color_features[6]
    features['HS_Correlation'] = color_features[7]
    features['HV_Correlation'] = color_features[8]
    features['SV_Correlation'] = color_features[9]
    features['Skin_Consistency'] = color_features[10]
    
    # Extract compression features
    _, compr_features = extract_compression_features(img, mask)
    features['DCT_Mean'] = compr_features[0]
    features['DCT_Std'] = compr_features[1]
    features['DCT_Range'] = compr_features[2]
    features['Variance_Mean'] = compr_features[3]
    features['Variance_Std'] = compr_features[4]
    features['Energy_Ratio'] = compr_features[5]
    
    return features

def extract_landmark_features(img, mask):
    """Extract facial landmark features from an image"""
    # Apply mask to focus only on face
    masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    
    # Convert to grayscale if not already
    if len(masked_img.shape) > 2:
        masked_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        masked_gray = masked_img.copy()
    
    # Create face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(masked_gray, 1.1, 4)
    
    # Initialize visualization and feature vectors
    landmark_viz = masked_img.copy()
    landmark_features = []
    
    # If no face detected in the masked area, use the entire mask as the ROI
    if len(faces) == 0:
        # Get mask contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            faces = np.array([[x, y, w, h]])
    
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Use face geometry for feature extraction
            face_roi = masked_gray[y:y+h, x:x+w]
            
            # Extract key regions based on facial proportions
            eye_region_h = int(h * 0.25)
            eye_region_w = int(w * 0.3)
            eye_y = y + int(h * 0.25)
            left_eye_x = x + int(w * 0.2)
            right_eye_x = x + int(w * 0.5)
            
            # Extract mouth region
            mouth_y = y + int(h * 0.65)
            mouth_h = int(h * 0.25)
            mouth_x = x + int(w * 0.25)
            mouth_w = int(w * 0.5)
            
            # Calculate facial proportions
            eye_distance = right_eye_x - left_eye_x
            face_width = w
            face_height = h
            eye_to_mouth = mouth_y - eye_y
            
            # Calculate important facial ratios
            eye_ratio = eye_distance / face_width
            height_width_ratio = face_height / face_width
            eye_mouth_ratio = eye_to_mouth / face_height
            
            # Add to feature vector
            landmark_features = [
                eye_ratio, 
                height_width_ratio,
                eye_mouth_ratio,
                eye_distance,
                face_width,
                face_height,
                eye_to_mouth
            ]
            
            # Only process the largest face
            break
    
    # If no faces found, return empty feature vector
    if not landmark_features:
        landmark_features = [0, 0, 0, 0, 0, 0, 0]
    
    # Convert to numpy array
    landmark_features = np.array(landmark_features)
    
    return landmark_viz, landmark_features

def extract_illumination_features(img, mask):
    """Extract illumination features from an image"""
    # Apply mask to focus only on face
    masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    
    # Convert to LAB color space for better lighting analysis
    lab_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2LAB)
    
    # Extract L channel (luminance)
    l_channel = lab_img[:,:,0]
    
    # Calculate illumination gradient (direction of lighting)
    grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate gradient magnitude and direction
    magnitude = cv2.magnitude(grad_x, grad_y)
    angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    # Normalize magnitude for visualization
    norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Find specular highlights (brightest regions)
    thresh_value = np.percentile(l_channel[mask > 0], 95)  # Adaptive threshold for highlights
    _, highlights = cv2.threshold(l_channel, thresh_value, 255, cv2.THRESH_BINARY)
    highlights = cv2.bitwise_and(highlights, mask)
    
    # Analyze shadow regions (darkest areas)
    shadow_thresh = np.percentile(l_channel[mask > 0], 20)  # Adaptive threshold for shadows
    _, shadows = cv2.threshold(l_channel, shadow_thresh, 255, cv2.THRESH_BINARY_INV)
    shadows = cv2.bitwise_and(shadows, mask)
    
    # Calculate analytical metrics
    valid_pixels = mask > 0
    
    if np.sum(valid_pixels) > 0:
        # Direction consistency (lower std = more consistent lighting direction)
        direction_consistency = np.std(angle[valid_pixels])
        
        # Calculate highlight to shadow ratio
        highlight_pixels = np.sum(highlights > 0)
        shadow_pixels = np.sum(shadows > 0)
        highlight_shadow_ratio = highlight_pixels / max(shadow_pixels, 1)
        
        # Calculate local contrast within illumination
        local_contrast = np.std(l_channel[valid_pixels])
        
        # Calculate illumination uniformity
        illumination_uniformity = np.max(l_channel[valid_pixels]) - np.min(l_channel[valid_pixels])
        
        # Calculate average gradient magnitude
        avg_magnitude = np.mean(magnitude[valid_pixels])
        
        # Calculate primary light direction
        sin_sum = np.sum(np.sin(np.radians(angle[valid_pixels])) * magnitude[valid_pixels])
        cos_sum = np.sum(np.cos(np.radians(angle[valid_pixels])) * magnitude[valid_pixels])
        primary_direction = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
    else:
        # Default values if no valid pixels
        direction_consistency = 0
        highlight_shadow_ratio = 0
        local_contrast = 0
        illumination_uniformity = 0
        avg_magnitude = 0
        primary_direction = 0
    
    # Compile illumination features
    illumination_features = np.array([
        primary_direction,
        direction_consistency,
        highlight_shadow_ratio,
        local_contrast,
        illumination_uniformity,
        avg_magnitude
    ])
    
    # Create a dummy visualization (not needed for batch processing)
    illumination_viz = masked_img.copy()
    
    return illumination_viz, illumination_features

def extract_color_features(img, mask):
    """Extract color features from an image"""
    # Apply mask to focus only on face
    masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    
    # Convert to HSV color space for better color analysis
    hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
    
    # Split channels
    h, s, v = cv2.split(hsv_img)
    
    # Calculate color statistics
    valid_pixels = mask > 0
    
    if np.sum(valid_pixels) > 0:
        # Hue statistics
        hue_mean = np.mean(h[valid_pixels])
        hue_std = np.std(h[valid_pixels])
        
        # Saturation statistics
        sat_mean = np.mean(s[valid_pixels])
        sat_std = np.std(s[valid_pixels])
        
        # Value statistics
        val_mean = np.mean(v[valid_pixels])
        val_std = np.std(v[valid_pixels])
        
        # Calculate color contrasts
        color_contrast = sat_std * hue_std
        
        # Calculate color uniformity (correlation between channels)
        h_flat = h[valid_pixels].flatten()
        s_flat = s[valid_pixels].flatten()
        v_flat = v[valid_pixels].flatten()
        
        hs_correlation = np.corrcoef(h_flat, s_flat)[0, 1] if len(h_flat) > 1 else 0
        hv_correlation = np.corrcoef(h_flat, v_flat)[0, 1] if len(h_flat) > 1 else 0
        sv_correlation = np.corrcoef(s_flat, v_flat)[0, 1] if len(h_flat) > 1 else 0
    else:
        hue_mean, hue_std = 0, 0
        sat_mean, sat_std = 0, 0
        val_mean, val_std = 0, 0
        color_contrast = 0
        hs_correlation, hv_correlation, sv_correlation = 0, 0, 0
    
    # Calculate skin tone consistency (using pre-defined skin tone ranges)
    skin_mask = np.zeros_like(mask)
    
    # Simple skin detection in HSV space
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask1 = cv2.inRange(hsv_img, lower_skin, upper_skin)
    
    lower_skin = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin = np.array([180, 255, 255], dtype=np.uint8)
    skin_mask2 = cv2.inRange(hsv_img, lower_skin, upper_skin)
    
    skin_mask = skin_mask1 + skin_mask2
    
    # Apply original mask to skin mask
    skin_mask = cv2.bitwise_and(skin_mask, mask)
    
    # Calculate skin tone consistency
    skin_pixels = np.sum(skin_mask > 0)
    if skin_pixels > 0:
        skin_consistency = skin_pixels / np.sum(valid_pixels)
    else:
        skin_consistency = 0
    
    # Compile color features
    color_features = np.array([
        hue_mean,
        hue_std,
        sat_mean, 
        sat_std,
        val_mean,
        val_std,
        color_contrast,
        hs_correlation,
        hv_correlation,
        sv_correlation,
        skin_consistency
    ])
    
    # Create a dummy visualization (not needed for batch processing)
    color_viz = masked_img.copy()
    
    return color_viz, color_features

def extract_compression_features(img, mask):
    """Extract compression features from an image"""
    # Apply mask to focus only on face
    masked_img = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    
    # Convert to grayscale if not already
    if len(masked_img.shape) > 2:
        masked_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    else:
        masked_gray = masked_img.copy()
    
    # Apply DCT transform (discrete cosine transform)
    # Split image into 8x8 blocks for DCT analysis (similar to JPEG)
    h, w = masked_gray.shape
    h_blocks = h // 8
    w_blocks = w // 8
    
    # Initialize feature arrays
    dct_energy = []
    block_variances = []
    
    # Process each 8x8 block
    for i in range(h_blocks):
        for j in range(w_blocks):
            # Get block
            block = masked_gray[i*8:(i+1)*8, j*8:(j+1)*8].astype(np.float32)
            
            # Check if block is part of the mask
            block_mask = mask[i*8:(i+1)*8, j*8:(j+1)*8]
            if np.sum(block_mask) < 32:  # Skip blocks with less than half pixels in mask
                continue
            
            # Apply DCT
            block_dct = cv2.dct(block)
            
            # Calculate DCT coefficients energy
            block_energy = np.sum(np.abs(block_dct))
            dct_energy.append(block_energy)
            
            # Calculate block variance
            block_var = np.var(block)
            block_variances.append(block_var)
    
    # Calculate compression features
    if dct_energy:
        dct_mean = np.mean(dct_energy)
        dct_std = np.std(dct_energy)
        dct_max = np.max(dct_energy)
        dct_min = np.min(dct_energy)
        dct_range = dct_max - dct_min
        
        var_mean = np.mean(block_variances)
        var_std = np.std(block_variances)
        
        # Calculate energy distribution (low vs high frequency)
        dct_energy_sorted = np.sort(dct_energy)
        low_freq_energy = np.sum(dct_energy_sorted[:len(dct_energy_sorted)//2])
        high_freq_energy = np.sum(dct_energy_sorted[len(dct_energy_sorted)//2:])
        energy_ratio = high_freq_energy / (low_freq_energy + 1e-10)
    else:
        dct_mean, dct_std, dct_max, dct_min, dct_range = 0, 0, 0, 0, 0
        var_mean, var_std = 0, 0
        energy_ratio = 0
    
    # Compile compression features
    compression_features = np.array([
        dct_mean,
        dct_std,
        dct_range,
        var_mean,
        var_std,
        energy_ratio
    ])
    
    # Create a dummy visualization (not needed for batch processing)
    compression_viz = masked_img.copy()
    
    return compression_viz, compression_features

def process_dir(input_dir, output_dir):
    """
    Process all images in all subdirectories of input_dir and save feature CSVs
    
    Parameters:
    -----------
    input_dir : str
        Path to directory containing subdirectories with images
    output_dir : str
        Path to directory where CSV files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all subdirectories (IDs)
    id_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    print(f"Found {len(id_dirs)} ID directories to process.")
    
    # Process each ID directory
    for id_dir in tqdm(id_dirs, desc="Processing ID directories"):
        # Get ID from directory name
        id_name = os.path.basename(id_dir)
        
        # Get all images in this ID directory
        image_files = glob.glob(os.path.join(id_dir, "*.jpg")) + glob.glob(os.path.join(id_dir, "*.png"))
        
        if not image_files:
            print(f"No images found in {id_dir}, skipping.")
            continue
        
        print(f"Processing {len(image_files)} images for ID: {id_name}")
        
        # Initialize DataFrame to store features
        all_features = []
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Processing images for {id_name}", leave=False):
            # Get frame name from file path
            frame_name = os.path.basename(img_path)
            
            # Extract features
            features = extract_features_from_image(img_path)
            
            if features is None:
                print(f"Skipping image {img_path} due to extraction error")
                continue
            
            # Add frame name to features
            features['frame_name'] = frame_name
            
            # Add to list
            all_features.append(features)
        
        if not all_features:
            print(f"No features extracted for ID: {id_name}, skipping CSV creation")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Set frame_name as index
        df.set_index('frame_name', inplace=True)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f"{id_name}.csv")
        df.to_csv(csv_path)
        print(f"Saved features for ID {id_name} to {csv_path}")

def main():
    # Set input and output directories
    input_dir = "fake_158"  # Directory with all ID subdirectories
    output_dir = "manual_features_fake"  # Output directory for CSV files
    
    print(f"Starting feature extraction from {input_dir} to {output_dir}")
    
    # Process all directories
    process_dir(input_dir, output_dir)
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main()