import cv2
import numpy as np
import argparse
import sys
import os
from src.predictor import SolarPredictor
from src.geometry import analyze_buffers, sqft_to_sqmeters, area_to_radius_meters, BUFFER_2_SQFT
from src.image_loader import fetch_satellite_image

def main():
    parser = argparse.ArgumentParser(description="Solar Panel Buffer Detection CLI")
    
    # Input Group: Either Local Image OR Lat/Lon
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to local input image")
    group.add_argument("--lat", type=float, help="Latitude of the location")
    
    parser.add_argument("--lon", type=float, help="Longitude (required if --lat is used)")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    
    args = parser.parse_args()
    
    # Validate Lat/Lon pairing
    if args.lat and args.lon is None:
        parser.error("--lon is required when --lat is specified.")

    # 1. Load Image
    img = None
    scale = 0.15 # Default
    
    if args.image:
        print(f"Loading local image: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            print("Error: Could not read image file.")
            return
    else:
        # Fetch from Satellite
        print(f"Fetching satellite view for {args.lat}, {args.lon}...")
        
        # We fetch a larger context (50m radius = 100m wide) to help the model
        # The buffer logic deals with the specific 2400 sq.ft area later
        FETCH_RADIUS_METERS = 50.0 
        
        pil_img, fetched_scale = fetch_satellite_image(args.lat, args.lon, radius_m=FETCH_RADIUS_METERS)
        
        if pil_img:
            # Convert PIL to OpenCV (RGB -> BGR)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            scale = fetched_scale
            print(f"Image fetched successfully. Scale: {scale:.4f} m/px")
            
            # Save for debugging/record
            cv2.imwrite("fetched_result.jpg", img)
            print("Saved satellite view to fetched_result.jpg")
        else:
            print("Error: Failed to fetch satellite image.")
            return

    # 2. Prediction
    print(f"Loading model: {args.model}")
    try:
        predictor = SolarPredictor(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    panels = predictor.predict(img)
    print(f"Detected {len(panels)} panels.")
    
    # 3. Geometry Analysis
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2 # Target is always center for fetched images
    
    result = analyze_buffers(cx, cy, panels, scale)
    
    # 4. Report
    print("-" * 30)
    print(f"Status: {result['status']}")
    print(f"Total Area: {result['total_area_sqft']:.2f} sq.ft")
    print(f"Valid Panels: {len(result['valid_panels'])}")
    print("-" * 30)

if __name__ == "__main__":
    main()
