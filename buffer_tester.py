import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

# --- CONFIGURATION ---
MODEL_PATH = 'TRAINED_MODEL/best.pt'
IMAGE_FILE = 'test1.jpg'
OUTPUT_FILE = 'buffer_test_result.jpg'

# ASSUMPTION: Satellite Image Scale
# 0.15 meters per pixel is typical for high-res satellite (Type 1/Google Maps max zoom)
METERS_PER_PIXEL = 0.15  

# BUFFER DEFINITIONS (from FAQ)
BUFFER_1_SQFT = 1200
BUFFER_2_SQFT = 2400

def sqft_to_sqmeters(sqft):
    return sqft * 0.092903

def area_to_radius_meters(area_sqm):
    return math.sqrt(area_sqm / math.pi)

def meters_to_pixels(meters, scale):
    return meters / scale

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return
    if not os.path.exists(IMAGE_FILE):
        print(f"❌ Error: Image not found at {IMAGE_FILE}")
        return

    # Calculate Radii
    r1_meters = area_to_radius_meters(sqft_to_sqmeters(BUFFER_1_SQFT))
    r1_pixels = meters_to_pixels(r1_meters, METERS_PER_PIXEL)

    r2_meters = area_to_radius_meters(sqft_to_sqmeters(BUFFER_2_SQFT))
    r2_pixels = meters_to_pixels(r2_meters, METERS_PER_PIXEL)

    print(f"--- BUFFER CALCULATIONS ---")
    print(f"Scale: {METERS_PER_PIXEL} meters/pixel")
    print(f"Buffer 1 (1200 sq.ft): {r1_meters:.2f}m Radius -> {r1_pixels:.1f} Pixels")
    print(f"Buffer 2 (2400 sq.ft): {r2_meters:.2f}m Radius -> {r2_pixels:.1f} Pixels")

    # Load Model
    print(f"🧠 Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Load Image
    img_cv2 = cv2.imread(IMAGE_FILE)
    if img_cv2 is None:
        print("❌ Error: Could not read image.")
        return

    height, width = img_cv2.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    print(f"📸 Running prediction on {IMAGE_FILE} ({width}x{height})...")
    # Run inference
    results = model.predict(IMAGE_FILE, conf=0.15, verbose=False)
    result = results[0]

    # Extract Polygons
    panels = []
    if result.masks:
        for args in result.masks.xy:
            if len(args) >= 3:
                poly = Polygon(args)
                panels.append(poly)
    
    print(f"🔍 Detected {len(panels)} potential solar panels by YOLO.")

    # Create Buffer Zones (Shapely Geometry)
    center_point = Point(center_x, center_y)
    buffer_1_poly = center_point.buffer(r1_pixels)
    buffer_2_poly = center_point.buffer(r2_pixels)

    # --- CORE LOGIC START ---
    valid_panels = []
    zone_status = "No Solar in Buffer"
    active_buffer_pixels = r2_pixels # Default to outer for visualization
    check_poly = None

    # Logic: Check Buffer 1 FIRST
    in_buffer_1 = False
    for p in panels:
        if p.intersects(buffer_1_poly):
            in_buffer_1 = True
            break
    
    if in_buffer_1:
        zone_status = "Found in Buffer 1 (1200 sq.ft)"
        check_poly = buffer_1_poly
        active_buffer_pixels = r1_pixels
    else:
        # Check Buffer 2
        in_buffer_2 = False
        for p in panels:
            if p.intersects(buffer_2_poly):
                in_buffer_2 = True
                break
        
        if in_buffer_2:
            zone_status = "Found in Buffer 2 (2400 sq.ft)"
            check_poly = buffer_2_poly
            active_buffer_pixels = r2_pixels
        else:
            check_poly = None # No intersection found

    total_area_pixels = 0
    if check_poly:
        for p in panels:
            if p.intersects(check_poly):
                valid_panels.append(p)
                total_area_pixels += p.area

    total_area_sqm = total_area_pixels * (METERS_PER_PIXEL ** 2)
    total_area_sqft = total_area_sqm * 10.7639
    # --- CORE LOGIC END ---

    print(f"✅ RESULT: {zone_status}")
    print(f"   Valid Panels: {len(valid_panels)}")
    print(f"   Total Area: {total_area_sqft:.2f} sq.ft")

    # Visualization
    vis_img = img_cv2.copy()
    
    # Draw Buffers
    # Outer (2400) - Cyan
    cv2.circle(vis_img, (center_x, center_y), int(r2_pixels), (255, 255, 0), 2) 
    # Inner (1200) - Blue
    cv2.circle(vis_img, (center_x, center_y), int(r1_pixels), (255, 0, 0), 2)
    
    # Draw ALL YOLO Detections (Gray outlines)
    for p in panels:
        pts = np.array(p.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [pts], True, (100, 100, 100), 1)

    # Draw VALID Panels (Green Filled)
    overlay = vis_img.copy()
    for p in valid_panels:
        pts = np.array(p.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    
    # Blend overlay
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    # Add Text
    cv2.putText(vis_img, f"Status: {zone_status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(vis_img, f"Area: {total_area_sqft:.2f} sq.ft", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Save
    cv2.imwrite(OUTPUT_FILE, vis_img)
    print(f"🖼️ Saved visualization to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
