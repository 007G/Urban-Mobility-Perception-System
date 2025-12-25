"""
main.py - Main script for YOLOv8 detection with ByteTrack tracking
"""
import cv2
import numpy as np
from ultralytics import YOLO
from bytetrack import ByteTracker
import argparse
from pathlib import Path
import time


class VehiclePersonTracker:
    """Main class for vehicle and person detection and tracking with ByteTrack"""
    
    def __init__(self, model_path, track_thresh=0.5, track_buffer=50, match_thresh=0.8,
                 use_appearance=True):
        """
        Initialize detector and tracker
        
        Args:
            model_path: Path to YOLOv8 .pt model
            track_thresh: Confidence threshold for track initialization
            track_buffer: Number of frames to keep lost tracks (increased to 50)
            match_thresh: IOU threshold for matching
            use_appearance: Use appearance features for better re-ID
        """
        print(f"Loading YOLOv8 model: {model_path}")
        self.model = YOLO(model_path)
        
        # Initialize ByteTrack tracker with improved settings
        # track_buffer=50 allows objects to be tracked for 50 frames during occlusion
        self.tracker = ByteTracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            use_appearance=use_appearance
        )
        
        # Target classes: car, motorcycle, person (COCO dataset class IDs)
        self.target_classes = {
            0: 'person',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        # Colors for visualization (BGR format)
        self.colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'motorcycle': (0, 165, 255), # Orange
            'bus': (255, 255, 0),       # Cyan
            'truck': (128, 0, 128)      # Purple
        }
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'avg_fps': 0,
            'total_detections': 0,
            'unique_tracks': set()
        }
        
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process video with detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s")
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"  Output: {output_path}")
        
        frame_count = 0
        fps_list = []
        start_time = time.time()
        
        print("\nProcessing video...")
        print("-" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                frame_count += 1
                
                # Run detection (lower confidence to get more detections for ByteTrack)
                results = self.model(frame, conf=0.25, verbose=False)
                
                # Extract detections for target classes
                detections = self._extract_detections(results[0])
                
                # Update tracker (pass frame for appearance features)
                tracks = self.tracker.update(detections, frame)
                
                # Update statistics
                self.stats['total_detections'] += len(detections)
                for track in tracks:
                    self.stats['unique_tracks'].add(int(track[4]))
                
                # Draw results
                annotated_frame = self._draw_tracks(frame.copy(), tracks)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                fps_list.append(current_fps)
                
                # Add frame info
                info_text = [
                    f"Frame: {frame_count}/{total_frames}",
                    f"FPS: {current_fps:.1f}",
                    f"Active Tracks: {len(tracks)}",
                    f"Total Unique IDs: {len(self.stats['unique_tracks'])}"
                ]
                
                y_offset = 30
                for i, text in enumerate(info_text):
                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (5, y_offset + i * 30 - 20),
                        (15 + text_w, y_offset + i * 30 + 5),
                        (0, 0, 0),
                        -1
                    )
                    cv2.putText(
                        annotated_frame,
                        text,
                        (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                
                # Display frame
                if display:
                    cv2.imshow('ByteTrack - Vehicle & Person Tracking', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nProcessing interrupted by user")
                        break
                    elif key == ord('p'):
                        print("\nPaused. Press any key to continue...")
                        cv2.waitKey(0)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Progress update
                if frame_count % 30 == 0 or frame_count == total_frames:
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_count) * (total_frames - frame_count)
                    progress = (frame_count / total_frames) * 100
                    
                    print(f"Progress: {progress:5.1f}% | "
                          f"Frame: {frame_count:4d}/{total_frames} | "
                          f"FPS: {np.mean(fps_list[-30:]):.1f} | "
                          f"Tracks: {len(tracks):3d} | "
                          f"ETA: {eta:5.1f}s")
        
        except KeyboardInterrupt:
            print("\n\nProcessing interrupted by user (Ctrl+C)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        self.stats['total_frames'] = frame_count
        self.stats['avg_fps'] = np.mean(fps_list) if fps_list else 0
        
        print("\n" + "-" * 60)
        self._print_statistics()
    
    def _extract_detections(self, result):
        """Extract detections for target classes"""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                if cls_id in self.target_classes:
                    # Format: [x1, y1, x2, y2, conf, class_id]
                    detections.append([
                        box[0], box[1], box[2], box[3], conf, cls_id
                    ])
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def _draw_tracks(self, frame, tracks):
        """Draw tracked objects on frame"""
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = track
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            class_id = int(class_id)
            
            # Get class name and color
            class_name = self.target_classes.get(class_id, 'unknown')
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw bounding box with thicker line
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label background
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 15),
                (x1 + label_w + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
        
        return frame
    
    def _print_statistics(self):
        """Print tracking statistics"""
        print("\n" + "="*60)
        print("BYTETRACK STATISTICS")
        print("="*60)
        print(f"Total Frames Processed    : {self.stats['total_frames']}")
        print(f"Average Processing FPS    : {self.stats['avg_fps']:.2f}")
        print(f"Total Detections          : {self.stats['total_detections']}")
        print(f"Unique Track IDs          : {len(self.stats['unique_tracks'])}")
        
        if self.stats['total_frames'] > 0:
            avg_det = self.stats['total_detections'] / self.stats['total_frames']
            print(f"Avg Detections per Frame  : {avg_det:.2f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 + ByteTrack: Vehicle and Person Detection & Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --video input.mp4 --model yolov8n.pt
  
  # Save output video
  python main.py --video input.mp4 --model yolov8n.pt --output result.mp4
  
  # Tune ByteTrack parameters for better occlusion handling for 10 fps video
  python main.py --video input.mp4 --track-buffer 50 --track-thresh 0.4
  
  # Run without display (headless mode)
  python main.py --video input.mp4 --output result.mp4 --no-display
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Path to YOLOv8 model (.pt file). Default: yolov8n.pt'
    )
    
    # Output arguments
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output video (optional)'
    )
    
    # ByteTrack parameters
    parser.add_argument(
        '--track-thresh',
        type=float,
        default=0.5,
        help='Detection confidence threshold for track initialization. Default: 0.5'
    )
    parser.add_argument(
        '--track-buffer',
        type=int,
        default=50,
        help='Number of frames to keep lost tracks (occlusion handling). Default: 50'
    )
    parser.add_argument(
        '--match-thresh',
        type=float,
        default=0.8,
        help='IOU threshold for matching detections to tracks. Default: 0.8'
    )
    
    # Display arguments
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display during processing'
    )
    parser.add_argument(
        '--no-appearance',
        action='store_true',
        help='Disable appearance features (faster but less accurate re-ID)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.video).exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Print configuration
    print("\n" + "="*60)
    print("BYTETRACK VEHICLE & PERSON TRACKER")
    print("="*60)
    print(f"Video Input       : {args.video}")
    print(f"Model             : {args.model}")
    print(f"Output            : {args.output if args.output else 'None (display only)'}")
    print("\nByteTrack Configuration:")
    print(f"  Track Threshold : {args.track_thresh} (high-conf detection threshold)")
    print(f"  Track Buffer    : {args.track_buffer} frames (occlusion handling)")
    print(f"  Match Threshold : {args.match_thresh} (IOU matching)")
    print(f"  Appearance      : {'Enabled' if not args.no_appearance else 'Disabled'}")
    print(f"  Display         : {'Disabled' if args.no_display else 'Enabled'}")
    print("="*60)
    
    # Initialize tracker
    tracker = VehiclePersonTracker(
        model_path=args.model,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        use_appearance=not args.no_appearance
    )
    
    # Process video
    tracker.process_video(
        video_path=args.video,
        output_path=args.output,
        display=not args.no_display
    )
    
    print("\n✓ Processing complete!\n")


if __name__ == '__main__':
    main()
