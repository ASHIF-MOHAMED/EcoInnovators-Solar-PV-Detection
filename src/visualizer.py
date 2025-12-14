import cv2
import numpy as np
import math
from shapely.geometry import Point

class BufferVisualizer:
    """
    Visualizes buffer zones and detected solar panels on satellite imagery.
    """
    
    def __init__(self):
        # Colors (BGR format for OpenCV)
        self.color_buffer_1 = (0, 255, 255)    # Yellow for 1200 sqft buffer
        self.color_buffer_2 = (255, 0, 255)    # Magenta for 2400 sqft buffer
        self.color_panel_valid = (0, 255, 0)   # Green for valid panels
        self.color_panel_invalid = (0, 0, 255) # Red for invalid panels
        self.color_center = (255, 255, 255)    # White for center point
    
    def sqft_to_radius_pixels(self, area_sqft, scale):
        """
        Convert area in square feet to radius in pixels.
        
        Args:
            area_sqft: Area in square feet
            scale: meters per pixel
        
        Returns:
            radius in pixels
        """
        area_sqm = area_sqft * 0.092903  # Convert sqft to sqm
        radius_m = math.sqrt(area_sqm / math.pi)
        radius_px = radius_m / scale
        return int(radius_px)
    
    def draw_buffer_zones(self, image, cx, cy, scale, buffer_1_sqft=1200, buffer_2_sqft=2400):
        """
        Draw buffer zones on the image.
        
        Args:
            image: Input image (will be modified)
            cx, cy: Center coordinates
            scale: meters per pixel
            buffer_1_sqft: First buffer area in sqft
            buffer_2_sqft: Second buffer area in sqft
        
        Returns:
            Modified image
        """
        # Calculate radii
        r1 = self.sqft_to_radius_pixels(buffer_1_sqft, scale)
        r2 = self.sqft_to_radius_pixels(buffer_2_sqft, scale)
        
        # Draw buffer circles
        cv2.circle(image, (cx, cy), r2, self.color_buffer_2, 3)  # Outer buffer (2400 sqft)
        cv2.circle(image, (cx, cy), r1, self.color_buffer_1, 3)  # Inner buffer (1200 sqft)
        
        # Draw center point
        cv2.circle(image, (cx, cy), 8, self.color_center, -1)
        cv2.circle(image, (cx, cy), 8, (0, 0, 0), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Buffer 1: {buffer_1_sqft} sqft", 
                    (cx + r1 + 10, cy), font, 0.6, self.color_buffer_1, 2)
        cv2.putText(image, f"Buffer 2: {buffer_2_sqft} sqft", 
                    (cx + r2 + 10, cy + 30), font, 0.6, self.color_buffer_2, 2)
        
        return image
    
    def draw_panels(self, image, panels, cx, cy, scale, valid_panels=None):
        """
        Draw detected solar panels on the image.
        
        Args:
            image: Input image (will be modified)
            panels: List of Shapely Polygon objects
            cx, cy: Center coordinates
            scale: meters per pixel
            valid_panels: List of panel indices that are valid (inside buffer)
        
        Returns:
            Modified image
        """
        if valid_panels is None:
            valid_panels = []
        
        for idx, panel in enumerate(panels):
            # Extract polygon from dictionary if needed
            polygon = panel['polygon'] if isinstance(panel, dict) else panel
            
            # Get polygon coordinates
            coords = np.array(polygon.exterior.coords, dtype=np.int32)
            
            # Determine color based on validity
            if idx in valid_panels:
                color = self.color_panel_valid
                thickness = 3
            else:
                color = self.color_panel_invalid
                thickness = 2
            
            # Draw polygon
            cv2.polylines(image, [coords], True, color, thickness)
            
            # Fill with semi-transparent color
            overlay = image.copy()
            cv2.fillPoly(overlay, [coords], color)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Add panel number with confidence
            centroid = polygon.centroid
            label_pos = (int(centroid.x), int(centroid.y))
            
            # Add confidence if available
            if isinstance(panel, dict) and 'confidence' in panel:
                label = f"P{idx+1}:{panel['confidence']:.2f}"
            else:
                label = f"P{idx+1}"
            
            cv2.putText(image, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def add_info_panel(self, image, result, metrics=None):
        """
        Add information panel with detection results.
        
        Args:
            image: Input image (will be modified)
            result: Detection result dictionary
            metrics: Quality metrics dictionary
        
        Returns:
            Modified image
        """
        h, w = image.shape[:2]
        
        # Create semi-transparent background for info panel
        panel_height = 200 if metrics else 150
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (500, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 35
        line_height = 30
        
        # Title
        cv2.putText(image, "Solar PV Detection Results", 
                   (20, y_offset), font, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # Status
        status_color = (0, 255, 0) if result['qc_status'] == 'VERIFIABLE' else (0, 0, 255)
        cv2.putText(image, f"Status: {result['status']}", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # QC Status
        cv2.putText(image, f"QC: {result['qc_status']}", 
                   (20, y_offset), font, 0.5, status_color, 2)
        y_offset += line_height
        
        # Area
        cv2.putText(image, f"Area: {result['total_area_sqft']:.2f} sqft", 
                   (20, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += line_height
        
        # Zone ID
        if result.get('zone_id') is not None:
            cv2.putText(image, f"Zone: {result['zone_id']}", 
                       (20, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += line_height
        
        # Quality metrics (if provided)
        if metrics:
            cv2.putText(image, f"Quality: Cloud={metrics.get('cloud_coverage', 0):.1%} " +
                       f"Blur={metrics.get('blur_score', 0):.0f}", 
                       (20, y_offset), font, 0.4, (200, 200, 200), 1)
        
        return image
    
    def create_visualization(self, image, cx, cy, panels, scale, result, 
                           valid_panel_indices=None, quality_metrics=None):
        """
        Create complete visualization with all elements.
        
        Args:
            image: Input image
            cx, cy: Center coordinates
            panels: List of detected panels
            scale: meters per pixel
            result: Detection result dictionary
            valid_panel_indices: List of valid panel indices
            quality_metrics: Quality metrics dictionary
        
        Returns:
            Visualization image
        """
        # Create a copy to avoid modifying original
        vis_image = image.copy()
        
        # Draw buffer zones
        vis_image = self.draw_buffer_zones(vis_image, cx, cy, scale)
        
        # Draw panels
        if panels:
            vis_image = self.draw_panels(vis_image, panels, cx, cy, scale, valid_panel_indices)
        
        # Add info panel
        vis_image = self.add_info_panel(vis_image, result, quality_metrics)
        
        # Add legend
        vis_image = self._add_legend(vis_image)
        
        return vis_image
    
    def _add_legend(self, image):
        """Add color legend to the image."""
        h, w = image.shape[:2]
        
        # Position legend at bottom right
        x_start = w - 250
        y_start = h - 150
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (x_start, y_start), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = y_start + 25
        
        cv2.putText(image, "Legend:", (x_start + 10, y), font, 0.5, (255, 255, 255), 1)
        y += 25
        
        # Buffer 1
        cv2.circle(image, (x_start + 20, y), 5, self.color_buffer_1, -1)
        cv2.putText(image, "Buffer 1 (1200 sqft)", (x_start + 35, y + 5), 
                   font, 0.4, (255, 255, 255), 1)
        y += 25
        
        # Buffer 2
        cv2.circle(image, (x_start + 20, y), 5, self.color_buffer_2, -1)
        cv2.putText(image, "Buffer 2 (2400 sqft)", (x_start + 35, y + 5), 
                   font, 0.4, (255, 255, 255), 1)
        y += 25
        
        # Valid panels
        cv2.rectangle(image, (x_start + 15, y - 5), (x_start + 25, y + 5), 
                     self.color_panel_valid, -1)
        cv2.putText(image, "Valid Panel", (x_start + 35, y + 5), 
                   font, 0.4, (255, 255, 255), 1)
        y += 25
        
        # Invalid panels
        cv2.rectangle(image, (x_start + 15, y - 5), (x_start + 25, y + 5), 
                     self.color_panel_invalid, -1)
        cv2.putText(image, "Invalid Panel", (x_start + 35, y + 5), 
                   font, 0.4, (255, 255, 255), 1)
        
        return image
