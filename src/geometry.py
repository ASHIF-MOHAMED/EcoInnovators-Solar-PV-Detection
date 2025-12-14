import math
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon

BUFFER_1_SQFT = 1200
BUFFER_2_SQFT = 2400

METERS_PER_PIXEL_DEFAULT = 0.15

def sqft_to_sqmeters(sqft):
    """Converts square feet to square meters."""
    return sqft * 0.092903

def area_to_radius_meters(area_sqm):
    """Calculates radius of a circle given its area."""
    return math.sqrt(area_sqm / math.pi)

def get_buffer_radii(scale_mpp=METERS_PER_PIXEL_DEFAULT):
    """
    Returns the pixel radii for Buffer 1 and Buffer 2 based on the image scale.
    """
    r1_m = area_to_radius_meters(sqft_to_sqmeters(BUFFER_1_SQFT))
    r2_m = area_to_radius_meters(sqft_to_sqmeters(BUFFER_2_SQFT))
    
    return {
        "buffer_1_px": r1_m / scale_mpp,
        "buffer_2_px": r2_m / scale_mpp,
        "buffer_1_m": r1_m,
        "buffer_2_m": r2_m
    }

def determine_qc_status(
    detected,
    total_area_sqft,
    zone_id,
    cloud_coverage=0.0,
    blur_score=0.0,
    brightness=0.0,
    contrast=0.0
):
    """
    Determine QC status based on image quality and detection.
    
    VERIFIABLE (present): Clear evidence of solar
    VERIFIABLE (not present): Clear evidence of NO solar
    NOT_VERIFIABLE (reason): Insufficient evidence (shadow, cloud, blur, etc)
    """
    
   
    if cloud_coverage > 0.3:
        return "NOT_VERIFIABLE (cloud_coverage)"
    
    if blur_score < 50:
        return "NOT_VERIFIABLE (blur)"
    
    if brightness < 20 or brightness > 250:
        return "NOT_VERIFIABLE (brightness)"
    
    if contrast < 30:
        return "NOT_VERIFIABLE (contrast)"
    
    if detected:
        return "VERIFIABLE (present)"
    else:
        return "VERIFIABLE (not present)"

def merge_panels(panels):
    """
    Merges overlapping panel geometries to avoid double counting.
    Returns a single geometry (Polygon or MultiPolygon) or None.
    """
    if not panels:
        return None
    return unary_union(panels)

def analyze_buffers(center_x, center_y, panels, scale_mpp=METERS_PER_PIXEL_DEFAULT, 
                     cloud_coverage=0.0, blur_score=100.0, brightness=50.0, contrast=50.0):
    radii = get_buffer_radii(scale_mpp)
    center = Point(center_x, center_y)

    buffer1_poly = center.buffer(radii["buffer_1_px"])
    buffer2_poly = center.buffer(radii["buffer_2_px"])

    zone_detected = 0
    valid_panels = []

    panel_polygons = [p['polygon'] if isinstance(p, dict) else p for p in panels]
    
    detected_b1 = [p for p in panel_polygons if p.intersects(buffer1_poly)]
    detected_b2 = [p for p in panel_polygons if p.intersects(buffer2_poly)]

    if detected_b1:
        valid_panels = detected_b1
        zone_detected = 1
        status = "Found in Buffer 1 (High Confidence)"
    elif detected_b2:
        valid_panels = detected_b2
        zone_detected = 2
        status = "Found in Buffer 2 (Expanded Search)"
    else:
        status = "No Solar Detected"

    merged_panels = merge_panels(valid_panels)

    if merged_panels:
        total_px_area = merged_panels.area
        total_sqm = total_px_area * (scale_mpp ** 2)
        total_sqft = total_sqm * 10.7639
    else:
        total_sqft = 0

    qc_status = determine_qc_status(
        detected=bool(valid_panels),
        total_area_sqft=total_sqft,
        zone_id=zone_detected,
        cloud_coverage=cloud_coverage,
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast
    )

    valid_panel_indices = []
    valid_panel_areas_sqm = []
    polygon_masks = []
    
    if valid_panels:
        for panel in valid_panels:
            if panel in panel_polygons:
                idx = panel_polygons.index(panel)
                valid_panel_indices.append(idx)
                
                panel_area_px = panel.area
                panel_area_sqm = panel_area_px * (scale_mpp ** 2)
                valid_panel_areas_sqm.append(panel_area_sqm)
                
                coords = [[int(x), int(y)] for x, y in panel.exterior.coords[:-1]]
                polygon_masks.append(coords)
    
    panel_count = len(valid_panels)
    avg_panel_area_sqm = sum(valid_panel_areas_sqm) / panel_count if panel_count > 0 else 0.0
    max_panel_area_sqm = max(valid_panel_areas_sqm) if valid_panel_areas_sqm else 0.0
    min_panel_area_sqm = min(valid_panel_areas_sqm) if valid_panel_areas_sqm else 0.0
    
    return {
        "status": status,
        "zone_id": zone_detected,
        "qc_status": qc_status,
        "total_area_sqft": total_sqft,
        "total_area_sqm": total_sqft * 0.092903,
        "panel_count": panel_count,
        "avg_panel_area_sqm": avg_panel_area_sqm,
        "max_panel_area_sqm": max_panel_area_sqm,
        "min_panel_area_sqm": min_panel_area_sqm,
        "valid_panel_indices": valid_panel_indices,
        "valid_panel_areas_sqm": valid_panel_areas_sqm,
        "polygon_masks": polygon_masks,
        "geometry": {
            "center": (center_x, center_y),
            "buffer_1": buffer1_poly,
            "buffer_2": buffer2_poly,
            "merged_panels": merged_panels
        }
    }

