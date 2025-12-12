import math
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

def analyze_buffers(center_x, center_y, panels, scale_mpp=METERS_PER_PIXEL_DEFAULT):
    """
    Core business logic: Checks intersection of panels with buffer zones.
    Returns the status, valid panels, and total area.
    """
    radii = get_buffer_radii(scale_mpp)
    center = Point(center_x, center_y)
    
    buffer1_poly = center.buffer(radii["buffer_1_px"])
    buffer2_poly = center.buffer(radii["buffer_2_px"])
    
    status = "No Solar in Buffer"
    valid_panels = []
    zone_detected = 0 # 0=None, 1=Buffer1, 2=Buffer2

    # 1. Check Buffer 1 (Priority)
    detected_in_b1 = [p for p in panels if p.intersects(buffer1_poly)]
    
    if detected_in_b1:
        status = "Found in Buffer 1 (1200 sq.ft)"
        valid_panels = detected_in_b1
        zone_detected = 1
    else:
        # 2. Check Buffer 2
        detected_in_b2 = [p for p in panels if p.intersects(buffer2_poly)]
        if detected_in_b2:
            status = "Found in Buffer 2 (2400 sq.ft)"
            valid_panels = detected_in_b2
            zone_detected = 2
            
    # Calculate Total Area
    total_px_area = sum(p.area for p in valid_panels)
    total_sqm = total_px_area * (scale_mpp ** 2)
    total_sqft = total_sqm * 10.7639
    
    return {
        "status": status,
        "zone_id": zone_detected,
        "valid_panels": valid_panels,
        "total_area_sqft": total_sqft,
        "geometry": {
            "center": (center_x, center_y),
            "r1": radii["buffer_1_px"],
            "r2": radii["buffer_2_px"]
        }
    }
