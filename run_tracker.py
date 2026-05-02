import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import DronePersonDetector
from src.drone_tracker import DroneByteTracker
from src.visualizer import TrackVisualizer
from src.visdrone_loader import VisDroneLoader, VisDroneSequence, VideoLoader


def process_sequence(
    source,
    detector: DronePersonDetector,
    tracker: DroneByteTracker,
    visualizer: TrackVisualizer,
    output_path: str,
    mot_output_path: str = None,
    show: bool = False,
    max_frames: int = 0,
):
    """Process a single sequence or video.
    
    Args:
        source: VisDroneSequence or VideoLoader
        detector: Person detector
        tracker: ByteTrack tracker
        visualizer: Visualization renderer
        output_path: Output video file path
        show: Show live window
        max_frames: Max frames to process (0 = all)
    """
    # Determine source properties
    if isinstance(source, VisDroneSequence):
        w, h = source.width, source.height
        source_name = source.name
        n_total = source.n_frames
        src_fps = 30.0
    elif isinstance(source, VideoLoader):
        w, h = source.width, source.height
        source_name = Path(source.path).stem
        n_total = source.n_frames
        src_fps = source.fps
    else:
        raise ValueError(f"Unknown source type: {type(source)}")
    
    # Create output video writer
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    writer = visualizer.create_video_writer(output_path, w, h, fps=src_fps)
    
    # Reset tracker for new sequence
    tracker.reset()
    
    logger.info(f"Processing: {source_name} ({n_total} frames, {w}x{h})")
    logger.info(f"Output: {output_path}")
    
    # Timing
    total_time = 0.0
    frame_count = 0
    all_fps = []
    mot_results = []  # Collect MOT-format results for evaluation
    
    for frame_id, frame in source:
        if max_frames > 0 and frame_count >= max_frames:
            break
        
        t_start = time.perf_counter()
        
        # ---- Detection ----
        detections = detector.detect(frame)
        
        # ---- Tracking (with CMC) ----
        tracks = tracker.update(frame, detections)
        
        # Record for post-processing interpolation
        tracker.record_for_interpolation(frame_id, tracks)
        
        # ---- Timing ----
        elapsed = time.perf_counter() - t_start
        total_time += elapsed
        frame_count += 1
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        all_fps.append(current_fps)
        
        # ---- Collect MOT-format results ----
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            w, h = x2 - x1, y2 - y1
            mot_results.append(
                f"{frame_id},{track.track_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{track.confidence:.4f},-1,-1,-1"
            )
        
        # ---- Visualization ----
        extra_stats = {
            'Det': f'{detector.avg_ms:.0f}ms',
            'Trk': f'{tracker.avg_tracking_ms:.1f}ms',
        }
        if tracker.cmc:
            extra_stats['CMC'] = f'{tracker.cmc.success_rate*100:.0f}%'
            # Show EMAT motion severity
            motion = tracker.cmc.motion_severity()
            severity = motion['severity']
            severity_bar = '█' * int(severity * 10) + '░' * (10 - int(severity * 10))
            extra_stats['Motion'] = f'{severity_bar} {severity:.2f}'
        
        annotated = visualizer.draw(
            frame, tracks,
            fps=current_fps,
            frame_id=frame_id,
            det_count=len(detections),
            extra_stats=extra_stats,
        )
        
        writer.write(annotated)
        
        # Live display
        if show:
            cv2.imshow('Aerial Guardian', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("User quit")
                break
            elif key == ord('p'):
                cv2.waitKey(0)  # Pause
        
        # Progress logging
        if frame_count % 50 == 0 or frame_count == 1:
            avg_fps = frame_count / total_time if total_time > 0 else 0
            logger.info(
                f"  Frame {frame_id}/{n_total} | "
                f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f}) | "
                f"Tracks: {len(tracks)} | Dets: {len(detections)}"
            )
    
    writer.release()
    if show:
        cv2.destroyAllWindows()
    
    # Final stats
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    stats = {
        'sequence': source_name,
        'frames_processed': frame_count,
        'total_time_s': round(total_time, 2),
        'avg_fps': round(avg_fps, 1),
        'min_fps': round(min(all_fps) if all_fps else 0, 1),
        'max_fps': round(max(all_fps) if all_fps else 0, 1),
        'total_tracks': tracker.next_id - 1,
        'detector_stats': detector.get_stats(),
        'tracker_stats': tracker.get_stats(),
    }
    
    logger.info("=" * 60)
    logger.info(f"RESULTS: {source_name}")
    logger.info(f"  Frames: {frame_count} | Time: {total_time:.1f}s")
    logger.info(f"  FPS: {avg_fps:.1f} avg ({min(all_fps):.1f}-{max(all_fps):.1f})")
    logger.info(f"  Total tracks: {tracker.next_id - 1}")
    logger.info(f"  Detection: {detector.avg_ms:.1f}ms avg")
    logger.info(f"  Tracking: {tracker.avg_tracking_ms:.1f}ms avg")
    if tracker.cmc:
        logger.info(f"  CMC success: {tracker.cmc.success_rate*100:.1f}%")
    logger.info(f"  Output: {output_path}")
    
    # Save MOT-format results (with interpolation)
    if mot_output_path:
        # Post-processing: fill gaps via linear interpolation
        interpolated = tracker.interpolate_tracks(max_gap=5)
        for frame_id_interp, interp_list in interpolated.items():
            for track_id, bbox, conf in interp_list:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                mot_results.append(
                    f"{frame_id_interp},{track_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{conf:.4f},-1,-1,-1"
                )
        
        # Sort by frame_id for proper evaluation
        mot_results.sort(key=lambda x: int(x.split(',')[0]))
        
        os.makedirs(os.path.dirname(mot_output_path) if os.path.dirname(mot_output_path) else '.', exist_ok=True)
        with open(mot_output_path, 'w') as f:
            f.write('\n'.join(mot_results))
        logger.info(f"  MOT results: {mot_output_path} ({len(mot_results)} detections)")
    
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Aerial Guardian — Drone Person Detection + MOT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tracker.py --input data/VisDrone2019-MOT-val/sequences/uav0000013_00000_v
  python run_tracker.py --input data/VisDrone2019-MOT-val --all-sequences
  python run_tracker.py --input video.mp4 --sahi --show
        """
    )
    
    # Input/Output
    parser.add_argument('--input', required=True, help='Path to VisDrone sequence folder, dataset root, or video file')
    parser.add_argument('--output', default='output/', help='Output video path or directory')
    parser.add_argument('--all-sequences', action='store_true', help='Process all sequences in dataset')
    
    # Detection
    parser.add_argument('--model', default='yolo11n.pt', help='YOLO model path')
    parser.add_argument('--imgsz', type=int, default=1280, help='Detection input resolution')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--sahi', action='store_true', help='Enable SAHI tiled inference')
    parser.add_argument('--sahi-slice', type=int, default=640, help='SAHI tile size')
    parser.add_argument('--sahi-overlap', type=float, default=0.2, help='SAHI tile overlap ratio')
    
    # Tracking
    parser.add_argument('--no-cmc', action='store_true', help='Disable Camera Motion Compensation')
    parser.add_argument('--max-age', type=int, default=50, help='Max frames to keep lost track')
    parser.add_argument('--iou-thresh', type=float, default=0.3, help='IoU threshold for association')
    
    # Display
    parser.add_argument('--show', action='store_true', help='Show live visualization window')
    parser.add_argument('--max-frames', type=int, default=0, help='Max frames to process (0=all)')
    parser.add_argument('--no-trail', action='store_true', help='Disable trajectory tails')
    parser.add_argument('--save-mot', action='store_true', help='Save MOT-format .txt results for evaluation')
    parser.add_argument('--mot-dir', default='output/mot_results', help='Directory for MOT-format results')
    
    # Device
    parser.add_argument('--device', default='', help='CUDA device (e.g. "0" or "cpu")')
    
    args = parser.parse_args()
    
    # ---- Initialize pipeline components ----
    logger.info("=" * 60)
    logger.info("  AERIAL GUARDIAN — Drone Person MOT Pipeline")
    logger.info("=" * 60)
    
    # Detector
    detector = DronePersonDetector(
        model_path=args.model,
        imgsz=args.imgsz,
        confidence=args.conf,
        device=args.device,
        use_sahi=args.sahi,
        sahi_slice_size=args.sahi_slice,
        sahi_overlap=args.sahi_overlap,
    )
    
    # Tracker
    tracker = DroneByteTracker(
        use_cmc=not args.no_cmc,
        iou_threshold=args.iou_thresh,
        max_age=args.max_age,
    )
    
    # Visualizer
    visualizer = TrackVisualizer(
        show_trail=not args.no_trail,
    )
    
    logger.info(f"CMC: {'ENABLED' if not args.no_cmc else 'DISABLED'}")
    logger.info(f"SAHI: {'ENABLED' if args.sahi else 'DISABLED'}")
    
    input_path = Path(args.input)
    all_stats = []
    
    # ---- Determine input type and process ----
    if input_path.suffix in ['.mp4', '.avi', '.mov', '.mkv']:
        # Single video file
        source = VideoLoader(str(input_path))
        out_path = args.output if args.output.endswith('.mp4') else str(Path(args.output) / f"{input_path.stem}_tracked.mp4")
        mot_path = str(Path(args.mot_dir) / f"{input_path.stem}.txt") if args.save_mot else None
        stats = process_sequence(source, detector, tracker, visualizer, out_path, mot_path, args.show, args.max_frames)
        all_stats.append(stats)
        source.release()
        
    elif args.all_sequences:
        # All sequences in dataset
        loader = VisDroneLoader(str(input_path))
        
        if len(loader) == 0:
            logger.error(f"No sequences found in {input_path}")
            return
        
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for seq in loader:
            out_path = str(out_dir / f"{seq.name}_tracked.mp4")
            mot_path = str(Path(args.mot_dir) / f"{seq.name}.txt") if args.save_mot else None
            stats = process_sequence(seq, detector, tracker, visualizer, out_path, mot_path, args.show, args.max_frames)
            all_stats.append(stats)
        
        # Summary across all sequences
        total_frames = sum(s['frames_processed'] for s in all_stats)
        total_time = sum(s['total_time_s'] for s in all_stats)
        overall_fps = total_frames / total_time if total_time > 0 else 0
        
        logger.info("\n" + "=" * 60)
        logger.info("  OVERALL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Sequences processed: {len(all_stats)}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Overall FPS: {overall_fps:.1f}")
        for s in all_stats:
            logger.info(f"    {s['sequence']}: {s['avg_fps']:.1f} FPS, {s['total_tracks']} tracks")
        logger.info("=" * 60)
        
    else:
        # Single sequence folder
        # Check if it's a sequence folder (contains images) or dataset root
        if any(input_path.glob("*.jpg")) or any(input_path.glob("*.png")):
            # Single sequence
            ann_path = input_path.parent.parent / "annotations" / f"{input_path.name}.txt"
            source = VisDroneSequence(str(input_path), str(ann_path) if ann_path.exists() else None)
            out_path = args.output if args.output.endswith('.mp4') else str(Path(args.output) / f"{input_path.name}_tracked.mp4")
            mot_path = str(Path(args.mot_dir) / f"{input_path.name}.txt") if args.save_mot else None
            stats = process_sequence(source, detector, tracker, visualizer, out_path, mot_path, args.show, args.max_frames)
            all_stats.append(stats)
        else:
            # Try as dataset root — process first sequence
            loader = VisDroneLoader(str(input_path))
            if len(loader) == 0:
                logger.error(f"No sequences or images found in {input_path}")
                return
            
            seq = loader[0]
            out_path = args.output if args.output.endswith('.mp4') else str(Path(args.output) / f"{seq.name}_tracked.mp4")
            mot_path = str(Path(args.mot_dir) / f"{seq.name}.txt") if args.save_mot else None
            stats = process_sequence(seq, detector, tracker, visualizer, out_path, mot_path, args.show, args.max_frames)
            all_stats.append(stats)
            logger.info(f"Processed first sequence. Use --all-sequences to process all {len(loader)} sequences.")


if __name__ == '__main__':
    main()
