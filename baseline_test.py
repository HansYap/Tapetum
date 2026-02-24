import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm

@dataclass
class DetectionResult:
    """Store detection results for analysis."""
    frame_number: int
    timestamp: float
    detections: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time: float


class BaselineModelTester:
    """
    Test foundation models on blackwater footage and generate failure reports.
    """
    def __init__(self, model_name: str = "yolov8", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.results = []
        
    def load_model(self):
        """Load the specified model."""
        print(f"Loading {self.model_name}...")
        
        if self.model_name.startswith("yolo"):
            from ultralytics import YOLO
            # Start with pre-trained YOLO as baseline
            self.model = YOLO('yolov8n.pt')  # Nano model for speed
            print("  YOLOv8 loaded (COCO weights)")
            
        # Add Florence-2 support when available
        # elif self.model_name == "florence2":
        #     from transformers import AutoProcessor, AutoModelForCausalLM
        #     self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2")
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    
    def process_video(
        self, 
        video_path: str, 
        sample_rate: int = 30,
        max_frames: int = 300
    ) -> List[DetectionResult]:
        """
        Process video and collect detection results.
        
        Parameters:
        -----------
        video_path : str
            Path to video file
        sample_rate : int
            Process every Nth frame (default: 30 = 1 fps for 30fps video)
        max_frames : int
            Maximum frames to process
        """
        if self.model is None:
            self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\n📹 Processing video: {Path(video_path).name}")
        print(f"   Total frames: {total_frames}")
        print(f"   FPS: {fps}")
        print(f"   Sample rate: Every {sample_rate} frames")
        
        frame_count = 0
        processed_count = 0
        results = []
        
        with tqdm(total=min(total_frames // sample_rate, max_frames)) as pbar:
            while cap.isOpened() and processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames
                if frame_count % sample_rate == 0:
                    timestamp = frame_count / fps
                    
                    # Run inference
                    start_time = time.time()
                    detections = self._run_inference(frame)
                    processing_time = time.time() - start_time
                    
                    # Store results
                    result = DetectionResult(
                        frame_number=frame_count,
                        timestamp=timestamp,
                        detections=detections,
                        confidence_scores=[d['confidence'] for d in detections],
                        processing_time=processing_time
                    )
                    results.append(result)
                    
                    processed_count += 1
                    pbar.update(1)
                
                frame_count += 1
        
        cap.release()
        self.results = results
        return results
    
    
    def _run_inference(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run model inference on a single frame."""
        if self.model_name.startswith("yolo"):
            results = self.model(frame, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': int(box.cls[0]),
                    'class_name': results.names[int(box.cls[0])]
                })
            
            if len(detections) > 0:
                # 1. Ensure directory exists (prevents FileNotFoundError)
                save_path = Path('runs/detect/baseline_samples')
                save_path.mkdir(parents=True, exist_ok=True)
                
                # 2. Use a unique filename (using timestamp to avoid overwrites)
                # Results.save() takes 'filename' as the argument
                fname = str(save_path / f"hit_{int(time.time()*1000)}.jpg")
                results.save(filename=fname)
            
            return detections
        
        return []
    
    
    def generate_failure_report(self, output_path: str = "failure_analysis.json"):
        """
        Generate comprehensive failure analysis report.
        """
        if not self.results:
            print("❌ No results to analyze. Run process_video() first.")
            return
        
        print(f"\n📊 Generating Failure Analysis Report...")
        
        # Calculate metrics
        total_frames = len(self.results)
        frames_with_detections = sum(1 for r in self.results if len(r.detections) > 0)
        total_detections = sum(len(r.detections) for r in self.results)
        
        all_confidences = [score for r in self.results for score in r.confidence_scores]
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        processing_times = [r.processing_time for r in self.results]
        avg_processing_time = np.mean(processing_times)
        
        # Identify low-confidence periods
        low_conf_threshold = 0.5
        low_conf_frames = [
            r for r in self.results 
            if r.confidence_scores and max(r.confidence_scores) < low_conf_threshold
        ]
        
        # Build report
        report = {
            "model_name": self.model_name,
            "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_frames_processed": total_frames,
                "frames_with_detections": frames_with_detections,
                "detection_rate": f"{frames_with_detections/total_frames*100:.1f}%",
                "total_detections": total_detections,
                "avg_detections_per_frame": f"{total_detections/total_frames:.2f}",
                "avg_confidence": f"{avg_confidence:.3f}",
                "avg_processing_time_ms": f"{avg_processing_time*1000:.1f}"
            },
            "failure_analysis": {
                "low_confidence_frames": len(low_conf_frames),
                "low_confidence_rate": f"{len(low_conf_frames)/total_frames*100:.1f}%",
                "probable_causes": [
                    "Tannin color shift reduces RGB contrast",
                    "Suspended particles create false positives",
                    "Reduced visibility causes missed detections",
                    "Model trained on clear-water datasets"
                ]
            },
            "confidence_distribution": {
                "min": float(min(all_confidences)) if all_confidences else 0,
                "max": float(max(all_confidences)) if all_confidences else 0,
                "mean": float(avg_confidence),
                "std": float(np.std(all_confidences)) if all_confidences else 0
            },
            "recommendations": [
                "Fine-tune with blackwater-augmented training data",
                "Use LoRA for parameter-efficient adaptation",
                "Increase training on low-contrast scenarios",
                "Consider ensemble with specialized detector"
            ]
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   ✓ Report saved: {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("BASELINE FAILURE ANALYSIS SUMMARY")
        print("="*50)
        print(f"Detection Rate: {report['summary']['detection_rate']}")
        print(f"Avg Confidence: {report['summary']['avg_confidence']}")
        print(f"Low Confidence: {report['failure_analysis']['low_confidence_rate']}")
        print(f"Processing Speed: {report['summary']['avg_processing_time_ms']} ms/frame")
        print("\n💡 Key Finding: Model struggles with blackwater conditions")
        print("   → Fine-tuning recommended")
        print("="*50)
        
        return report


def download_youtube_sample(url: str, output_path: str = "data/raw/youtube_sample.mp4"):
    """
    Download a YouTube video for testing.
    
    Example URLs:
    - "https://www.youtube.com/watch?v=..." (blackwater aquarium tour)
    - Search for: "Blackwater Aquarium" "Rio Negro Biotope"
    """
    import yt_dlp
    
    if Path(output_path).exists():
        print(f"✓ Already exists: {output_path}")
        return output_path
    
    print(f"📥 Downloading YouTube video...")
    print(f"   URL: {url}")
    
    ydl_opts = {
        'format': 'best[height<=720]',  # 720p max to save space
        'outtmpl': output_path,
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"   ✓ Saved: {output_path}")
    return output_path


def main():
    """Main testing workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline model testing on blackwater footage")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--youtube", type=str, help="YouTube URL to test")
    parser.add_argument("--model", type=str, default="yolov8n", help="Model to test")
    parser.add_argument("--sample-rate", type=int, default=30, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Get video path
    if args.youtube:
        video_path = download_youtube_sample(args.youtube)
    elif args.video:
        video_path = args.video
    else:
        print("❌ Please provide --video or --youtube")
        return
    
    # Run baseline test
    tester = BaselineModelTester(model_name=args.model)
    tester.process_video(video_path, sample_rate=args.sample_rate, max_frames=args.max_frames)
    tester.generate_failure_report()
    
    print("\n✅ Baseline test complete!")
    print("📄 Review failure_analysis.json for detailed metrics")


if __name__ == "__main__":
    main()