"""
Analyze satellite image to determine zoom level based on visual features.
"""

import cv2
import numpy as np
import math

def estimate_zoom_level(image_path):
    """
    Estimate the zoom level of a satellite image based on visual characteristics.
    
    Zoom level estimation based on:
    - Object sizes (buildings, roads, vehicles)
    - Ground Sample Distance (GSD) - meters per pixel
    - Visual detail level
    
    Google Maps Zoom Levels:
    - Zoom 15: ~4.77 m/px (neighborhood view)
    - Zoom 16: ~2.39 m/px (street view)
    - Zoom 17: ~1.19 m/px (building details)
    - Zoom 18: ~0.60 m/px (clear building features)
    - Zoom 19: ~0.30 m/px (individual objects visible)
    - Zoom 20: ~0.15 m/px (very high detail)
    - Zoom 21: ~0.07 m/px (maximum detail)
    """
    
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image")
        return None
    
    h, w = img.shape[:2]
    print(f"Image dimensions: {w} x {h} pixels")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Analyze edge density (more edges = higher zoom)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)
    print(f"Edge density: {edge_density:.4f}")
    
    # 2. Analyze texture complexity
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = laplacian.var()
    print(f"Texture variance: {texture_variance:.2f}")
    
    # 3. Detect building-like structures
    # Use contour detection to find rectangular structures
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count significant rectangular structures
    building_count = 0
    building_sizes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small noise
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if roughly rectangular (4-6 vertices)
            if 4 <= len(approx) <= 6:
                building_count += 1
                building_sizes.append(math.sqrt(area))
    
    print(f"Detected building-like structures: {building_count}")
    
    if building_sizes:
        avg_building_size = np.mean(building_sizes)
        print(f"Average building size: {avg_building_size:.1f} pixels")
    else:
        avg_building_size = 0
    
    # 4. Estimate zoom level based on features
    estimated_zoom = None
    estimated_gsd = None
    
    # High detail indicators
    if edge_density > 0.15 and texture_variance > 200:
        # Very high detail - likely zoom 19-21
        if avg_building_size > 80:
            estimated_zoom = 20
            estimated_gsd = 0.15
            detail_level = "Very High (individual objects clearly visible)"
        elif avg_building_size > 40:
            estimated_zoom = 19
            estimated_gsd = 0.30
            detail_level = "High (building features clear)"
        else:
            estimated_zoom = 18
            estimated_gsd = 0.60
            detail_level = "Medium-High (buildings distinguishable)"
    
    elif edge_density > 0.08 and texture_variance > 100:
        # Medium detail - likely zoom 17-18
        estimated_zoom = 17
        estimated_gsd = 1.19
        detail_level = "Medium (neighborhood level)"
    
    else:
        # Lower detail - zoom 15-16
        estimated_zoom = 16
        estimated_gsd = 2.39
        detail_level = "Low-Medium (street level)"
    
    print("\n" + "="*60)
    print("ZOOM LEVEL ESTIMATION")
    print("="*60)
    print(f"Estimated Zoom Level: {estimated_zoom}")
    print(f"Estimated GSD (meters/pixel): {estimated_gsd}")
    print(f"Detail Level: {detail_level}")
    print("="*60)
    
    # Visual analysis notes
    print("\nVisual Characteristics:")
    print(f"- Buildings are {'clearly visible' if building_count > 5 else 'partially visible'}")
    print(f"- Edge detail: {'High' if edge_density > 0.15 else 'Medium' if edge_density > 0.08 else 'Low'}")
    print(f"- Texture complexity: {'High' if texture_variance > 200 else 'Medium' if texture_variance > 100 else 'Low'}")
    
    # Recommendation
    print("\n📍 Recommendation:")
    if estimated_zoom >= 19:
        print("   This image has high detail suitable for solar panel detection.")
        print("   Individual panels and rooftop features should be visible.")
    elif estimated_zoom >= 17:
        print("   This image has moderate detail for solar panel detection.")
        print("   Larger solar installations should be detectable.")
    else:
        print("   This image may have insufficient detail for accurate detection.")
        print("   Consider using a higher zoom level (19-20) for better results.")
    
    return {
        "estimated_zoom": estimated_zoom,
        "estimated_gsd": estimated_gsd,
        "edge_density": edge_density,
        "texture_variance": texture_variance,
        "building_count": building_count,
        "avg_building_size": avg_building_size
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_zoom_level.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = estimate_zoom_level(image_path)
