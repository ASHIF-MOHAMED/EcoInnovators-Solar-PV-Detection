import cv2
import argparse
import sys
from src.predictor import SolarPredictor
from src.geometry import analyze_buffers

def main():
    parser = argparse.ArgumentParser(description="Solar Panel Buffer Detection CLI")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--scale", type=float, default=0.15, help="Meters per pixel (default: 0.15)")
    
    args = parser.parse_args()
    
    # 1. Prediction
    print(f"Loading model: {args.model}")
    try:
        predictor = SolarPredictor(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Processing image: {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        print("Error: Could not read image.")
        return
        
    panels = predictor.predict(img)
    print(f"Detected {len(panels)} panels.")
    
    # 2. Geometry Analysis
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    result = analyze_buffers(cx, cy, panels, args.scale)
    
    # 3. Report
    print("-" * 30)
    print(f"Status: {result['status']}")
    print(f"Total Area: {result['total_area_sqft']:.2f} sq.ft")
    print(f"Valid Panels: {len(result['valid_panels'])}")
    print("-" * 30)

if __name__ == "__main__":
    main()
