import cv2
import numpy as np
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BlackwaterAugmentation:
    """
    simulate blackwater conditions
    Parameters:
    -----------
    tannin_intensity : float (0.0-1.0)
        How dark/brown the water appears (0=clear, 1=heavy tannin)
    turbidity : float (0.0-1.0)
        Amount of suspended particles/cloudiness
    contrast_reduction : float (0.0-1.0)
        How much contrast is lost due to water scattering
    """
    def __init__(
        self, 
        tannin_intensity: float = 0.6, 
        turbidity: float = 0.3, 
        contrast_reduction: float = 0.4, 
        add_particles: bool = True,
    ):
        self.tannin_intensity = tannin_intensity
        self.turbidity = turbidity
        self.contrast_reduction = contrast_reduction
        self.add_particles = add_particles
    
    
    def apply_tannin_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply color shift to simulate blackwater colour"""
        
        img_float = image.astype(np.float32) / 255.0
        
        tannin_r = 1.0 - (self.tannin_intensity * 0.35)  # Slight red reduction
        tannin_g = 1.0 - (self.tannin_intensity * 0.55)  # More green absorption
        tannin_b = 1.0 - (self.tannin_intensity * 0.65)  # Heavy blue absorptio
        
        # opencv/numpy store images format: (height, width, channels)
        img_float[:, :, 0] = img_float[:, :, 0] * tannin_b  # Blue channel
        img_float[:, :, 1] = img_float[:, :, 1] * tannin_g  # Green channel
        img_float[:, :, 2] = img_float[:, :, 2] * tannin_r  # Red channel
        
        # Add ambient brown/amber glow
        amber_overlay = np.full_like(img_float, [0.15, 0.28, 0.42])  # RGB
        img_float = cv2.addWeighted(
            img_float, 1.0, 
            amber_overlay, self.tannin_intensity * 0.6, 
            0
        )
        
        return np.clip(img_float * 255, 0, 255).astype(np.uint8)
    
    
    def apply_turbidity(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce contrast and add fog effect to simulate suspended particles.
        """
         # Reduce overall contrast
        alpha = 1.0 - (self.contrast_reduction * 0.5)
        beta = 40 * self.contrast_reduction
        
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        fog = np.ones_like(image) * (25 + 35 * self.turbidity)
        fogged = cv2.addWeighted(
            adjusted, 1.0 - (self.turbidity * 0.7),
            fog.astype(np.uint8), self.turbidity * 0.7,
            0
        )
        
        return fogged
    
    
    def add_suspended_particles(self, image: np.ndarray, num_particles: int = 150) -> np.ndarray:
        """
        Add white/bright particles to simulate mulm, detritus, and organic matter.
        """
        if not self.add_particles:
            return image
        
        img_copy = image.copy()
        h, w = image.shape[:2]
        
        actual_particles = int(num_particles * (0.5 + self.turbidity))
        
        for _ in range(actual_particles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            # Particle size (smaller particles more common)
            size = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])
            
            # Particle brightness (whitish to light brown)
            brightness = np.random.randint(180, 255)
            color = (brightness, brightness, int(brightness * 0.9))
            
            # Draw particle with slight blur
            cv2.circle(img_copy, (x, y), size, color, -1)
        
        # Slight blur to make particles look more natural
        img_copy = cv2.GaussianBlur(img_copy, (3, 3), 0)
        
        return img_copy
    
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full blackwater transformation pipeline.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,2] = cv2.add(lab[:,:,2], int(15 * self.tannin_intensity))  # shift toward red/yellow
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Step 1: Tannin color shift
        transformed = self.apply_tannin_filter(image)
        
        hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 - self.turbidity*0.3), 0, 255)
        transformed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Step 2: Turbidity and contrast reduction
        transformed = self.apply_turbidity(transformed)
        
        h, w = transformed.shape[:2]
        depth = np.linspace(0, 1, h).reshape(h,1,1)
        blur = cv2.GaussianBlur(transformed, (5,5), 0)
        transformed = (transformed*(1-depth*self.turbidity) + blur*(depth*self.turbidity)).astype(np.uint8)
        
        # Step 3: Add suspended particles
        transformed = self.add_suspended_particles(transformed)
        
        return transformed

def create_training_augmentation(image_size: Tuple[int, int] = (640, 640)):
    """
    Create Albumentations pipeline for model training.
    Includes both blackwater simulation AND standard augmentations.
    """
    
    blackwater_transform = A.Lambda(
        image=BlackwaterAugmentation(
            tannin_intensity=np.random.uniform(0.3, 0.8),
            turbidity=np.random.uniform(0.2, 0.6),
            contrast_reduction=np.random.uniform(0.3, 0.6)
        )
    )
    
    return A.Compose([
        # Resize
        A.Resize(height=image_size[0], width=image_size[1]),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=10, p=0.5),
        
        # Apply blackwater effect
        blackwater_transform,
        
        # Additional lighting variations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
        
        # Normalize for model input
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    
def create_validation_augmentation(image_size: Tuple[int, int] = (640, 640)):
    """
    Validation pipeline (no geometric augmentation, consistent blackwater).
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Lambda(image=BlackwaterAugmentation(
            tannin_intensity=0.6,
            turbidity=0.4,
            contrast_reduction=0.4
        )),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python blackwater_augmentation.py <input_image>")
        sys.exit(1)
    
    # Load image
    img = cv2.imread(sys.argv[1])
    
    if img is None:
        print(f"Error: Could not load image {sys.argv[1]}")
        sys.exit(1)
        
    
    # Create augmentor with different intensity levels
    intensities = [0.3, 0.5, 0.7]
    
    for i, intensity in enumerate(intensities):
        augmentor = BlackwaterAugmentation(
            tannin_intensity=intensity,
            turbidity=intensity * 0.6,
            contrast_reduction=intensity * 0.5
        )
        
        result = augmentor(img)
        output_path = f"blackwater_example_{i+1}_intensity_{intensity}.jpg"
        cv2.imwrite(output_path, result)
        print(f"✓ Saved: {output_path}")
    
    print("\n✅ Augmentation demo complete!")
    print("Compare the originals with the blackwater versions.")