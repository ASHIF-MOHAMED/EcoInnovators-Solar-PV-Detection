import os
import math
import requests
from io import BytesIO
from PIL import Image
from pyproj import Transformer, CRS

# Esri World Imagery (XYZ Tiles) - Much more reliable than Export API
TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

def sqm_from_sqft(sqft: float) -> float:
    return sqft * 0.092903

def circle_radius_from_area_m2(area_m2: float) -> float:
    return math.sqrt(area_m2 / math.pi)

def latlon_to_tile(lat, lon, zoom):
    """
    Converts lat/lon to Web Mercator tile coordinates.
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def tile_to_latlon(xtile, ytile, zoom):
    """
    Returns the NW corner of the tile.
    """
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def fetch_satellite_image(lat, lon, radius_m, meters_per_pixel=0.15, max_dim=2048):
    """
    Fetches satellite imagery by stitching XYZ tiles.
    """
    # 1. Determine Zoom Level based on desired resolution
    # Resolution (m/px) = 156543.03 * cos(lat) / (2^zoom)
    # 0.15 m/px ~= Zoom 19
    target_scale = meters_per_pixel
    lat_rad = math.radians(lat)
    # solve for zoom: scale = C * cos(lat) / 2^z  => 2^z = C * cos(lat) / scale
    zoom = int(math.log2(156543.03 * math.cos(lat_rad) / target_scale))
    zoom = min(zoom, 19) # Cap at 19 (standard max for most free layers)
    
    print(f"Fetching tiles at Zoom: {zoom}")

    # 2. Calculate Tile Coordinates
    xtile, ytile = latlon_to_tile(lat, lon, zoom)
    
    # We fetch a 3x3 grid centered on the target to ensure safe coverage
    # (Since the target might be at the edge of a tile)
    tiles = []
    min_x, max_x = xtile - 1, xtile + 1
    min_y, max_y = ytile - 1, ytile + 1
    
    stitched_w = (max_x - min_x + 1) * 256
    stitched_h = (max_y - min_y + 1) * 256
    canvas = Image.new("RGB", (stitched_w, stitched_h))
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            url = TILE_URL.format(x=x, y=y, z=zoom)
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                tile_img = Image.open(BytesIO(r.content)).convert("RGB")
                
                # Paste into canvas
                px = (x - min_x) * 256
                py = (y - min_y) * 256
                canvas.paste(tile_img, (px, py))
            except Exception as e:
                print(f"Failed to fetch tile {x},{y}: {e}")
                # Continue blank (better than crashing)

    # 3. Crop to the target area
    # We need to find where our Lat/Lon is on this huge canvas
    # Top-Left of the canvas corresponds to tile (min_x, min_y) NW corner
    tl_lat, tl_lon = tile_to_latlon(min_x, min_y, zoom)
    
    # We need pixel coordinates of our target Lat/Lon relative to TL
    # We use Web Mercator projection logic for pixel offsets
    # Global pixel coordinates:
    n = 2.0 ** zoom
    
    def get_global_px(clat, clon):
        x = (clon + 180.0) / 360.0 * n * 256
        sin_y = math.sin(math.radians(clat))
        # Clipping optimization
        sin_y = min(max(sin_y, -0.9999), 0.9999)
        y = (0.5 - math.log((1 + sin_y) / (1 - sin_y)) / (4 * math.pi)) * n * 256
        return x, y

    target_gx, target_gy = get_global_px(lat, lon)
    tl_gx, tl_gy = get_global_px(tl_lat, tl_lon)
    
    # Local pixel coordinates on canvas
    center_px_x = target_gx - tl_gx
    center_px_y = target_gy - tl_gy
    
    # Crop box size
    box_radius_px = int(radius_m * 2.0 / target_scale) # Radius * padding / scale
    crop_size = box_radius_px * 2
    
    left = int(center_px_x - crop_size // 2)
    top = int(center_px_y - crop_size // 2)
    right = left + crop_size
    bottom = top + crop_size
    
    final_img = canvas.crop((left, top, right, bottom))
    
    print(f"Stitched & Cropped: {final_img.size}")
    return final_img, target_scale
