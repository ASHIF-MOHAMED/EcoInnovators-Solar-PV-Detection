import cv2
import numpy as np

class ImageQualityChecker:
    """
    Checks image quality and cloud coverage to determine if an image is verifiable.
    """
    
    def __init__(self, cloud_threshold=0.3, blur_threshold=100):
        """
        Args:
            cloud_threshold: Maximum acceptable cloud coverage ratio (0-1)
            blur_threshold: Minimum acceptable blur score (Laplacian variance)
        """
        self.cloud_threshold = cloud_threshold
        self.blur_threshold = blur_threshold
    
    def check_cloud_coverage(self, image):
        """
        Detect cloud coverage in satellite imagery.
        Returns cloud coverage ratio (0-1).
        """
        # Convert to HSV for better cloud detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Clouds are typically bright white/gray
        # Define range for white/bright areas
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Create mask for potential cloud areas
        cloud_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Calculate cloud coverage ratio
        total_pixels = image.shape[0] * image.shape[1]
        cloud_pixels = np.sum(cloud_mask > 0)
        cloud_ratio = cloud_pixels / total_pixels
        
        return cloud_ratio
    
    def check_blur(self, image):
        """
        Detect image blur using Laplacian variance.
        Returns blur score (higher = sharper).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def check_brightness(self, image):
        """
        Check if image is too dark or too bright.
        Returns average brightness (0-255).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def check_contrast(self, image):
        """
        Check image contrast using standard deviation.
        Returns contrast score.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def is_verifiable(self, image):
        """
        Comprehensive check to determine if image is verifiable.
        
        Returns:
            tuple: (is_verifiable: bool, reason_code: str, metrics: dict)
        """
        metrics = {}
        
        # Check cloud coverage
        cloud_ratio = self.check_cloud_coverage(image)
        metrics['cloud_coverage'] = round(cloud_ratio, 3)
        
        if cloud_ratio > self.cloud_threshold:
            return False, "heavy_cloud", metrics
        
        # Check blur (indicates low resolution or poor image quality)
        blur_score = self.check_blur(image)
        metrics['blur_score'] = round(blur_score, 2)
        
        if blur_score < self.blur_threshold:
            return False, "low_resolution", metrics
        
        # Check brightness
        brightness = self.check_brightness(image)
        metrics['brightness'] = round(brightness, 2)
        
        if brightness < 30:
            return False, "heavy_shadow", metrics
        elif brightness > 240:
            return False, "overexposed", metrics
        
        # Check contrast
        contrast = self.check_contrast(image)
        metrics['contrast'] = round(contrast, 2)
        
        if contrast < 20:
            return False, "low_contrast", metrics
        
        # All checks passed
        return True, "clear_image", metrics
    
    def get_qc_status(self, image):
        """
        Get QC status string for the image.
        
        Returns:
            str: "VERIFIABLE" or "NOT_VERIFIABLE"
        """
        is_verifiable, _, _ = self.is_verifiable(image)
        return "VERIFIABLE" if is_verifiable else "NOT_VERIFIABLE"
