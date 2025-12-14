from ultralytics import YOLO
from shapely.geometry import Polygon

class SolarPredictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path_or_array, conf=0.15):
        """
        Runs YOLO inference and returns a list of dictionaries with Shapely Polygons and metadata.
        """
        results = self.model.predict(image_path_or_array, conf=conf, verbose=False)
        result = results[0]
        
        panels = []
        if result.masks:
            for i, pts in enumerate(result.masks.xy):
                if len(pts) >= 3:
                    panel_data = {
                        'polygon': Polygon(pts),
                        'confidence': float(result.boxes.conf[i]) if result.boxes else 0.0,
                        'class_id': int(result.boxes.cls[i]) if result.boxes else 0
                    }
                    panels.append(panel_data)
        return panels
