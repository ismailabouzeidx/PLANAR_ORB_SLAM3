#!/usr/bin/env python3
"""
Preprocess images to enhance ORB feature detection
Applies: CLAHE, sharpening, and optional denoising
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

def preprocess_image(img, use_clahe=True, use_sharpen=True, use_denoise=False):
    """
    Preprocess image for better feature detection
    
    Args:
        img: Input image (BGR)
        use_clahe: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        use_sharpen: Apply unsharp masking
        use_denoise: Apply denoising (slower but removes noise)
    
    Returns:
        Preprocessed image
    """
    result = img.copy()
    
    # Convert to LAB color space for better CLAHE on luminance
    if use_clahe:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Unsharp masking for sharpening
    if use_sharpen:
        gaussian = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = cv2.addWeighted(result, 1.5, gaussian, -0.5, 0)
    
    # Denoising (optional, slower)
    if use_denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    
    return result

def process_sequence(input_dir, output_dir, use_clahe=True, use_sharpen=True, use_denoise=False):
    """
    Process all images in a sequence
    
    Args:
        input_dir: Directory with original images
        output_dir: Directory to save preprocessed images
        use_clahe: Apply CLAHE
        use_sharpen: Apply sharpening
        use_denoise: Apply denoising
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PPM files
    image_files = sorted(input_path.glob("*.ppm"))
    
    if not image_files:
        print(f"No PPM files found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    print(f"  CLAHE: {use_clahe}")
    print(f"  Sharpening: {use_sharpen}")
    print(f"  Denoising: {use_denoise}")
    print()
    
    for i, img_path in enumerate(image_files):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Preprocess
        processed = preprocess_image(img, use_clahe, use_sharpen, use_denoise)
        
        # Save
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), processed)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images...")
    
    print(f"\nDone! Processed {len(image_files)} images to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 preprocess_images.py <input_dir> <output_dir> [--no-clahe] [--no-sharpen] [--denoise]")
        print("\nExample:")
        print("  python3 preprocess_images.py /home/ismo/data/ins-coi/C4_30/frames /home/ismo/data/ins-coi/C4_30/frames_processed")
        print("\nOptions:")
        print("  --no-clahe:    Disable CLAHE contrast enhancement")
        print("  --no-sharpen:   Disable sharpening")
        print("  --denoise:      Enable denoising (slower)")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    use_clahe = "--no-clahe" not in sys.argv
    use_sharpen = "--no-sharpen" not in sys.argv
    use_denoise = "--denoise" in sys.argv
    
    process_sequence(input_dir, output_dir, use_clahe, use_sharpen, use_denoise)


