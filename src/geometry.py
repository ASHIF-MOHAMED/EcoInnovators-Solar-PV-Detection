import math
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon

# FAQ Requirements
BUFFER_1_SQFT = 1200
BUFFER_2_SQFT = 2400

# Constants
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
    zone_id
):
    if not detected:
        return "NOT_VERIFIABLE"

    if total_area_sqft < 50:
        return "NOT_VERIFIABLE"

    if zone_id == 2:
        return "VERIFIABLE"  # weaker confidence, but acceptable

    return "VERIFIABLE"

def merge_panels(panels):
    """
    Merges overlapping panel geometries to avoid double counting.
    Returns a single geometry (Polygon or MultiPolygon) or None.
    """
    if not panels:
        return None
    return unary_union(panels)

def analyze_buffers(center_x, center_y, panels, scale_mpp=METERS_PER_PIXEL_DEFAULT):
    radii = get_buffer_radii(scale_mpp)
    center = Point(center_x, center_y)

    buffer1_poly = center.buffer(radii["buffer_1_px"])
    buffer2_poly = center.buffer(radii["buffer_2_px"])

    zone_detected = 0
    valid_panels = []

    # Extract polygons from panel dictionaries
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
        zone_id=zone_detected
    )

    # Get indices of valid panels
    valid_panel_indices = []
    if valid_panels:
        for panel in valid_panels:
            if panel in panel_polygons:
                valid_panel_indices.append(panel_polygons.index(panel))
    
    return {
        "status": status,
        "zone_id": zone_detected,
        "qc_status": qc_status,
        "total_area_sqft": total_sqft,
        "valid_panel_indices": valid_panel_indices,
        "geometry": {
            "center": (center_x, center_y),
            "buffer_1": buffer1_poly,
            "buffer_2": buffer2_poly,
            "merged_panels": merged_panels
        }
    }

