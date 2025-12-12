from ultralytics import YOLO
from shapely.geometry import Polygon

class SolarPredictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def predict(self, image_path_or_array, conf=0.15):
        """
        Runs YOLO inference and returns a list of Shapely Polygons.
        """
        results = self.model.predict(image_path_or_array, conf=conf, verbose=False)
        result = results[0]
        
        panels = []
        if result.masks:
            for pts in result.masks.xy:
                if len(pts) >= 3:
                    panels.append(Polygon(pts))
        return panels
