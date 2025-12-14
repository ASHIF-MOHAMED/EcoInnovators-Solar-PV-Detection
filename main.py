import os
import json
import cv2
import numpy as np
import argparse
from datetime import datetime
import pandas as pd
from openpyxl import load_workbook
from src.predictor import SolarPredictor
from src.image_loader import fetch_satellite_image
from src.quality_checker import ImageQualityChecker
from src.geometry import analyze_buffers
from src.visualizer import BufferVisualizer


MODEL_PATH = "TRAINED_MODEL/detection_model.pt"
INPUT_XLSX = "input_samples.xlsx"
OUTPUT_DIR = "batch_output"
FETCH_RADIUS_METERS = 50.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

def map_qc_base(qc_status_str):
    if qc_status_str and qc_status_str.startswith("VERIFIABLE"):
        return "VERIFIABLE"
    return "NOT_VERIFIABLE"


def process_sample(sample_id, lat, lon, predictor, quality_checker, visualizer):
    out = {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": False,
        "confidence": 0.0,
        "pv_area_sqm_est": 0.0,
        "buffer_radius_sqft": 2400,
        "qc_status": "NOT_VERIFIABLE",
        "bbox_or_mask": "segmentation_mask",
        "image_metadata": {
            "source": "",
            "capture_date": datetime.now().strftime("%Y-%m-%d")
        }
    }

   
    pil_img, scale = fetch_satellite_image(lat, lon, radius_m=FETCH_RADIUS_METERS)
    if pil_img is None:
        out["qc_status"] = "NOT_VERIFIABLE"
        out["image_metadata"]["source"] = "tile_fetch_failed"
        return out

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    out["image_metadata"]["source"] = "Google Maps"

    
    is_verifiable, reason, metrics = quality_checker.is_verifiable(img)

  
    if not is_verifiable:
        reason_map = {
            "heavy_cloud": "cloud_coverage",
            "low_resolution": "blur",
            "heavy_shadow": "brightness",
            "overexposed": "brightness",
            "low_contrast": "contrast"
        }
        qc_reason = reason_map.get(reason, "image_quality")
        out["qc_status"] = f"NOT_VERIFIABLE ({qc_reason})"
      
        return out

   
    panels = predictor.predict(img)
    confidences = [p.get('confidence', 0.0) for p in panels if isinstance(p, dict)]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

   
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    result = analyze_buffers(
        cx, cy, panels, scale,
        cloud_coverage=metrics.get('cloud_coverage', 0.0),
        blur_score=metrics.get('blur_score', 0.0),
        brightness=metrics.get('brightness', 0.0),
        contrast=metrics.get('contrast', 0.0)
    )

    vis = visualizer.create_visualization(
        image=img,
        cx=cx,
        cy=cy,
        panels=panels,
        scale=scale,
        result=result,
        valid_panel_indices=result.get('valid_panel_indices', []),
        quality_metrics=metrics
    )

    vis_filename = os.path.join(OUTPUT_DIR, f"viz_{sample_id}.jpg")
    cv2.imwrite(vis_filename, vis)

    out["has_solar"] = result['zone_id'] > 0
    out["confidence"] = round(avg_conf, 2)
    out["pv_area_sqm_est"] = round(result.get('total_area_sqm', 0.0), 2)
    out["buffer_radius_sqft"] = 1200 if result['zone_id'] == 1 else 2400
    out["qc_status"] = map_qc_base(result.get('qc_status'))

    polygon_masks = result.get('polygon_masks', [])
    out["bbox_or_mask"] = "segmentation_mask"

    out["image_metadata"]["capture_date"] = datetime.now().strftime("%Y-%m-%d")

    return out


def main():
    parser = argparse.ArgumentParser(description="Batch runner for solar verification (Excel input)")
    parser.add_argument("--input", "-i", default=INPUT_XLSX, help="Path to input .xlsx file")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help="Path to .pt model file")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="Directory to write outputs")
    parser.add_argument("--radius", "-r", type=float, default=FETCH_RADIUS_METERS, help="Fetch radius in meters for satellite image")
    args = parser.parse_args()

    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(input_path)
    elif ext in ['.csv', '.txt']:
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    os.makedirs(args.output, exist_ok=True)

    
    predictor = SolarPredictor(args.model)
    quality_checker = ImageQualityChecker(cloud_threshold=0.3, blur_threshold=100)
    visualizer = BufferVisualizer()

    results = []

    required_cols = ['sample_id', 'lat', 'lon']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    for _, row in df.iterrows():
        sample_id = row['sample_id']
        lat = row['lat']
        lon = row['lon']
        print(f"Processing {sample_id}: {lat},{lon}")
        try:
            out = process_sample(int(sample_id), float(lat), float(lon), predictor, quality_checker, visualizer)
            json_path = os.path.join(args.output, f"output_{sample_id}.json")
            with open(json_path, 'w') as f:
                json.dump(out, f, indent=2)
            results.append(json_path)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")

    print("Batch complete. Outputs:")
    for p in results:
        print(" -", p)


if __name__ == '__main__':
    main()
