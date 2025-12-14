import cv2
import numpy as np
import argparse
import sys
import os
import json
from datetime import datetime
from src.predictor import SolarPredictor
from src.geometry import analyze_buffers
from src.image_loader import fetch_satellite_image
from src.quality_checker import ImageQualityChecker
from src.visualizer import BufferVisualizer

def main():
    parser = argparse.ArgumentParser(description="Solar Panel Buffer Detection CLI")
    
    # Input Group: Either Local Image OR Lat/Lon
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to local input image")
    group.add_argument("--lat", type=float, help="Latitude of the location")
    
    parser.add_argument("--lon", type=float, help="Longitude (required if --lat is used)")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--sample-id", type=int, help="Sample ID for output (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Validate Lat/Lon pairing
    if args.lat and args.lon is None:
        parser.error("--lon is required when --lat is specified.")

    # Generate sample_id if not provided
    sample_id = args.sample_id if args.sample_id else int(datetime.now().timestamp())
    
    # Store coordinates
    latitude = args.lat if args.lat else None
    longitude = args.lon if args.lon else None

    # 1. Load Image
    img = None
    scale = 0.15 # Default
    image_source = None
    
    if args.image:
        print(f"Loading local image: {args.image}")
        img = cv2.imread(args.image)
        image_source = "local"
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
            image_source = "Google Maps"
            print(f"Image fetched successfully. Scale: {scale:.4f} m/px")
            
            # Save for debugging/record
            cv2.imwrite("fetched_result.jpg", img)
            print("Saved satellite view to fetched_result.jpg")
        else:
            print("Error: Failed to fetch satellite image.")
            return

    # 2. Image Quality Check
    print("Checking image quality...")
    quality_checker = ImageQualityChecker(cloud_threshold=0.3, blur_threshold=100)
    is_verifiable, reason, metrics = quality_checker.is_verifiable(img)
    
    print(f"Quality Check: {reason}")
    print(f"Metrics: {metrics}")
    
    if not is_verifiable:
        print(f"⚠️  Image marked as NOT_VERIFIABLE: {reason}")
        
        # Map reason codes to specific QC status reasons
        reason_map = {
            "heavy_cloud": "cloud_coverage",
            "low_resolution": "blur",
            "heavy_shadow": "brightness",
            "overexposed": "brightness",
            "low_contrast": "contrast"
        }
        qc_reason = reason_map.get(reason, "image_quality")
        
        # Extended output format for NOT_VERIFIABLE
        output = {
            "sample_id": sample_id,
            "lat": round(latitude, 4) if latitude else None,
            "lon": round(longitude, 4) if longitude else None,
            "has_solar": False,
            "confidence": 0.0,
            "pv_area_sqm_est": 0,
            "buffer_radius_sqft": 2400,
            "qc_status": f"NOT_VERIFIABLE ({qc_reason})",
            "bbox_or_mask": "N/A",
            "image_metadata": {
                "source": image_source,
                "capture_date": datetime.now().strftime("%Y-%m-%d")
            }
        }
        
        # Save result and exit
        with open("output_result.json", "w") as f:
            json.dump(output, f, indent=2)
        print("Saved results to output_result.json")
        print(f"\n📊 Output Format:")
        print(f"   sample_id: {sample_id}")
        print(f"   has_solar: False")
        print(f"   buffer_radius_sqft: 2400")
        print(f"   pv_area_sqm_est: 0 sqm")
        print(f"   qc_status: NOT_VERIFIABLE")
        print(f"   Reason: {reason}")
        return
    
    # 3. Prediction
    print(f"Loading model: {args.model}")
    try:
        predictor = SolarPredictor(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    panels = predictor.predict(img)
    print(f"Raw detections from model: {len(panels)}")
    
    # Calculate average confidence from detected panels
    avg_confidence = 0.0
    if panels:
        confidences = [p['confidence'] for p in panels if isinstance(p, dict)]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2 
    result = analyze_buffers(
        cx, cy, panels, scale,
        cloud_coverage=metrics.get('cloud_coverage', 0.0),
        blur_score=metrics.get('blur_score', 100.0),
        brightness=metrics.get('brightness', 50.0),
        contrast=metrics.get('contrast', 50.0)
    )
    
    print("Creating visualization...")
    visualizer = BufferVisualizer()
    vis_image = visualizer.create_visualization(
        image=img,
        cx=cx,
        cy=cy,
        panels=panels,
        scale=scale,
        result=result,
        valid_panel_indices=result.get('valid_panel_indices', []),
        quality_metrics=metrics
    )
    
    vis_filename = f"output_visualization_sample_{sample_id}.jpg"
    cv2.imwrite(vis_filename, vis_image)
    print(f"🖼️  Saved visualization to {vis_filename}")
    
    print("-" * 60)
    print(f"Status           : {result['status']}")
    print(f"QC Status        : {result['qc_status']}")
    print(f"Zone ID          : {result['zone_id']}")
    print(f"Panel Count      : {result['panel_count']}")
    print(f"Total Area (sqm) : {result['total_area_sqm']:.2f} sqm")
    print(f"Total Area (sqft): {result['total_area_sqft']:.2f} sqft")
    print("-" * 60)

    WATT_PEAK_PER_M2 = 150
    capacity_kw = (result['total_area_sqm'] * WATT_PEAK_PER_M2) / 1000
    
    reason_code = "no_solar_detected"
    if result['panel_count'] > 0:
        if result['panel_count'] >= 3 and result['zone_id'] == 1:
            reason_code = "clear_array"
        elif result['panel_count'] >= 2:
            reason_code = "rectilinear_array"
        elif result['panel_count'] == 1:
            reason_code = "single_panel"
        elif avg_confidence < 0.5:
            reason_code = "low_confidence"
    
    if avg_confidence > 0.7:
        confidence_level = "HIGH"
    elif avg_confidence > 0.3:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"
    
    polygon_masks = result.get('polygon_masks', [])
    
    bbox_or_mask = "polygon_mask"
        
    output = {
        "sample_id": sample_id,
        "lat": round(latitude, 4) if latitude else None,
        "lon": round(longitude, 4) if longitude else None,
        "has_solar": result['zone_id'] > 0,
        "confidence": round(avg_confidence, 2),
        "pv_area_sqm_est": round(result['total_area_sqm'], 2),
        "buffer_radius_sqft": 1200 if result['zone_id'] == 1 else 2400,
        "qc_status": result["qc_status"],
        "bbox_or_mask": bbox_or_mask,
        "image_metadata": {
            "source": image_source,
            "capture_date": datetime.now().strftime("%Y-%m-%d")
        }
    }

    json_filename = f"output_result_sample_{sample_id}.json"
    with open(json_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {json_filename}")
    print(f"\n📊 Output Summary:")
    print(f"   sample_id: {sample_id}")
    print(f"   lat: {latitude}, lon: {longitude}")
    print(f"   has_solar: {result['zone_id'] > 0}")
    print(f"   panel_count: {result['panel_count']}")
    print(f"   pv_area_sqm: {result['total_area_sqm']:.2f}")
    print(f"   capacity_kw: {capacity_kw:.2f}")
    print(f"   confidence: {avg_confidence:.2f}")
    print(f"   reason_code: {reason_code}")
    print(f"   qc_status: {result['qc_status']}")
    print(f"   visualization: {vis_filename}")



if __name__ == "__main__":
    main()
