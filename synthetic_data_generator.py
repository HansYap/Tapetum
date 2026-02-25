import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import json
from tqdm import tqdm
import random
from blackwater_augmentation import BlackwaterAugmentation, create_training_augmentation

class SyntheticDatasetGenerator:
    """
    Generate synthetic blackwater training data from clear-water images.
    """
    
    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
     
        
    def generate_from_directory(
        self,
        source_dir: str,
        num_variations: int = 5,
        tannin_range: Tuple[float, float] = (0.55, 0.9)
    ):
        """
        Generate synthetic blackwater images from a directory of clear-water fish images.
        
        Parameters:
        -----------
        source_dir : str
            Directory containing source images
        num_variations : int
            How many blackwater variations to create per source image
        tannin_range : tuple
            (min, max) tannin intensity for random variation
        """
        source_path = Path(source_dir)
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
        
        if not image_files:
            print(f"❌ No images found in {source_dir}")
            return
        
        print(f"🎨 Generating synthetic blackwater dataset...")
        print(f"   Source images: {len(image_files)}")
        print(f"   Variations per image: {num_variations}")
        print(f"   Total outputs: {len(image_files) * num_variations}")
        
        dataset_info = {
            "source_directory": str(source_dir),
            "num_source_images": len(image_files),
            "num_variations": num_variations,
            "tannin_range": tannin_range,
            "generated_images": []
        }
        
        with tqdm(total=len(image_files) * num_variations) as pbar:
            for img_file in image_files:
                # Load source image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"⚠️  Could not load {img_file}")
                    continue
                
                # Generate variations
                for var_idx in range(num_variations):
                    # Random augmentation parameters
                    tannin = random.uniform(*tannin_range)
                    turbidity = random.uniform(0.35, 0.55)
                    contrast_reduction = random.uniform(0.1, 0.3)
                    
                    # Create augmentor
                    augmentor = BlackwaterAugmentation(
                        tannin_intensity=tannin,
                        turbidity=turbidity,
                        contrast_reduction=contrast_reduction,
                        add_particles=True
                    )
                    
                    # Apply transformation
                    augmented = augmentor(img)
                    
                    # Save
                    output_name = f"{img_file.stem}_var{var_idx:02d}.jpg"
                    output_path = self.images_dir / output_name
                    cv2.imwrite(str(output_path), augmented)
                    
                    # Record metadata
                    dataset_info["generated_images"].append({
                        "filename": output_name,
                        "source": img_file.name,
                        "tannin_intensity": tannin,
                        "turbidity": turbidity,
                        "contrast_reduction": contrast_reduction
                    })
                    
                    pbar.update(1)
        
        # Save dataset metadata
        metadata_path = self.output_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✅ Synthetic dataset generated!")
        print(f"   Images: {self.images_dir}")
        print(f"   Metadata: {metadata_path}")


    def create_mini_tank_dataset(
        self,
        video_path: str,
        sample_interval: int = 30,
        num_variations: int = 3
    ):
        """
        Extract frames from mini-tank video and create variations.
        
        Perfect for your "tea jar + toy fish" pilot experiment!
        
        Parameters:
        -----------
        video_path : str
            Path to mini-tank video
        sample_interval : int
            Extract every Nth frame
        num_variations : int
            Augmentation variations per frame
        """
        print(f"🔬 Creating mini-tank pilot dataset...")
        print(f"   Video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Total frames: {total_frames}")
        print(f"   Sampling every {sample_interval} frames")
        
        frame_count = 0
        extracted_count = 0
        
        with tqdm(total=total_frames // sample_interval) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % sample_interval == 0:
                    # Save original frame
                    frame_name = f"minitank_frame_{extracted_count:04d}.jpg"
                    cv2.imwrite(str(self.images_dir / frame_name), frame)
                    
                    # Create variations with different tannin levels
                    for var_idx in range(num_variations):
                        tannin = 0.3 + (var_idx * 0.2)  # 0.3, 0.5, 0.7
                        
                        augmentor = BlackwaterAugmentation(
                            tannin_intensity=tannin,
                            turbidity=tannin * 0.6,
                            contrast_reduction=tannin * 0.5
                        )
                        
                        augmented = augmentor(frame)
                        
                        var_name = f"minitank_frame_{extracted_count:04d}_var{var_idx}.jpg"
                        cv2.imwrite(str(self.images_dir / var_name), augmented)
                    
                    extracted_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        
        print(f"\n✅ Mini-tank dataset created!")
        print(f"   Frames extracted: {extracted_count}")
        print(f"   Total images: {extracted_count * (1 + num_variations)}")
        print(f"   Output: {self.images_dir}")
        
def main():
    """Command-line interface for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic blackwater datasets")
    parser.add_argument("--source-dir", type=str, help="Directory with clear-water fish images")
    parser.add_argument("--mini-tank-video", type=str, help="Mini-tank pilot video")
    parser.add_argument("--num-variations", type=int, default=5, help="Variations per image")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    
    args = parser.parse_args()

    
    generator = SyntheticDatasetGenerator(output_dir=args.output_dir)
    
    if args.source_dir:
        generator.generate_from_directory(
            source_dir=args.source_dir,
            num_variations=args.num_variations
        )
    
    elif args.mini_tank_video:
        generator.create_mini_tank_dataset(
            video_path=args.mini_tank_video,
            num_variations=3
        )
    
    else:
        print("❌ Please provide --source-dir or --mini-tank-video")
        print("   Or use --list-sources to see public dataset options")


if __name__ == "__main__":
    main()

        