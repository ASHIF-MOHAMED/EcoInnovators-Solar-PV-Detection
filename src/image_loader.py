import os
import math
import requests
from io import BytesIO
from PIL import Image
from pyproj import Transformer, CRS
from shapely.geometry import Point

ESRI_EXPORT_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

def sqm_from_sqft(sqft: float) -> float:
    return sqft * 0.092903

def circle_radius_from_area_m2(area_m2: float) -> float:
    return math.sqrt(area_m2 / math.pi)

def latlon_bbox_for_buffer(lat, lon, radius_m):
    aeqd_proj = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +units=m +datum=WGS84"
    aeqd = CRS.from_proj4(aeqd_proj)
    wgs84 = CRS.from_epsg(4326)
    
    # 3857 Transformer
    to_3857 = Transformer.from_crs(wgs84, CRS.from_epsg(3857), always_xy=True)
    
    # AEQD transformers
    to_aeqd = Transformer.from_crs(wgs84, aeqd, always_xy=True)
    to_wgs84 = Transformer.from_crs(aeqd, wgs84, always_xy=True)
    
    # Calc bbox in AEQD (meters from center)
    minx_m, miny_m = -radius_m, -radius_m
    maxx_m, maxy_m = radius_m, radius_m
    
    # Back to WGS84
    lon_min, lat_min = to_wgs84.transform(minx_m, miny_m)
    lon_max, lat_max = to_wgs84.transform(maxx_m, maxy_m)
    
    # Now project WGS84 bbox corners to 3857 for the request
    # Note: 3857 = Web Mercator (Meters)
    xmin_3857, ymin_3857 = to_3857.transform(lon_min, lat_min)
    xmax_3857, ymax_3857 = to_3857.transform(lon_max, lat_max)
    
    return (xmin_3857, ymin_3857, xmax_3857, ymax_3857), (lon_min, lat_min, lon_max, lat_max)

def fetch_satellite_image(lat, lon, radius_m, meters_per_pixel=0.15, max_dim=2048):
    radius_m = radius_m * 2.0 
    
    bbox_3857, bbox_4326 = latlon_bbox_for_buffer(lat, lon, radius_m)
    
    # Calculate pixel size based on 3857 meters
    w_meters = abs(bbox_3857[2] - bbox_3857[0])
    h_meters = abs(bbox_3857[3] - bbox_3857[1])
    
    width_px = int(w_meters / meters_per_pixel)
    height_px = int(h_meters / meters_per_pixel)
    
    width_px = max(100, width_px)
    height_px = max(100, height_px)

    print(f"BBox (3857): {bbox_3857}")
    print(f"Dimensions: {width_px}x{height_px}")
    
    # Use 3857 for SR
    params = {
        "bbox": f"{bbox_3857[0]},{bbox_3857[1]},{bbox_3857[2]},{bbox_3857[3]}",
        "bboxSR": "3857",
        "imageSR": "3857",
        "size": f"{width_px},{height_px}",
        "format": "png",
        "f": "image"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        r = requests.get(ESRI_EXPORT_URL, params=params, headers=headers, timeout=30)
        # print(f"DEBUG: {r.url}") 
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img, meters_per_pixel
    except Exception as e:
        print(f"Error: {e}")
        return None, None
