"""
DronePersonDetector — YOLO11n with Drone-Specific Adaptations
==============================================================
Aerial Guardian | VisDrone MOT Pipeline

Key adaptations over vanilla YOLO:
  1. Person-only filtering (COCO class 0 / VisDrone class 1)
  2. Altitude-adaptive confidence thresholding
  3. Optional SAHI tiled inference for ultra-small targets
  4. High-res input (1280px) for better small-object recall
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.error("ultralytics not installed. Run: pip install ultralytics")

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    logger.warning("SAHI not installed. Tiled inference disabled. Run: pip install sahi")


class DronePersonDetector:
    """YOLO-based person detector optimized for drone aerial footage.
    
    Drone-specific adaptations:
    - Altitude-adaptive confidence: lower threshold for small bboxes (<32px)
    - SAHI tiled inference: splits large frames into overlapping tiles
    - Person-only filtering: ignores all non-person detections
    - High-res mode: runs at 1280px for 2x better spatial resolution
    """
    
    # COCO person class ID
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        imgsz: int = 1280,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "",  # auto-select
        use_sahi: bool = False,
        sahi_slice_size: int = 640,
        sahi_overlap: float = 0.2,
        adaptive_conf: bool = True,
        small_obj_threshold: int = 32,  # pixels — bbox below this gets lower conf
        small_obj_conf: float = 0.15,   # lower conf for small objects
    ):
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.use_sahi = use_sahi and SAHI_AVAILABLE
        self.sahi_slice_size = sahi_slice_size
        self.sahi_overlap = sahi_overlap
        self.adaptive_conf = adaptive_conf
        self.small_obj_threshold = small_obj_threshold
        self.small_obj_conf = small_obj_conf
        self.device = device
        
        # Timing stats
        self.inference_times: List[float] = []
        self.frame_count = 0
        
        # Load YOLO model
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed")
        
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        logger.info(f"  ✓ Model loaded ({model_path})")
        logger.info(f"  ✓ Input size: {imgsz}px | Conf: {confidence} | SAHI: {use_sahi}")
        
        # SAHI model wrapper
        self.sahi_model = None
        if self.use_sahi:
            self._init_sahi(model_path)
    
    def _init_sahi(self, model_path: str):
        """Initialize SAHI detection model for tiled inference."""
        if not SAHI_AVAILABLE:
            return
        try:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',  # works for YOLO11 too
                model_path=model_path,
                confidence_threshold=self.confidence,
                device=self.device if self.device else 'cuda:0',
            )
            logger.info(f"  ✓ SAHI initialized: {self.sahi_slice_size}px tiles, "
                       f"{self.sahi_overlap*100:.0f}% overlap")
        except Exception as e:
            logger.error(f"SAHI init failed: {e}")
            self.sahi_model = None
            self.use_sahi = False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Detect persons in a drone frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            List of person detections:
            [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'label': 'person',
                'class_id': 0,
                'bbox_area': float,
                'is_small': bool,
            }, ...]
        """
        t0 = time.perf_counter()
        self.frame_count += 1
        
        if self.use_sahi and self.sahi_model is not None:
            detections = self._detect_sahi(frame)
        else:
            detections = self._detect_standard(frame)
        
        # Apply altitude-adaptive confidence filtering
        if self.adaptive_conf:
            detections = self._adaptive_filter(detections)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.inference_times.append(elapsed_ms)
        
        return detections
    
    def _detect_standard(self, frame: np.ndarray) -> List[Dict]:
        """Standard YOLO inference with person filtering."""
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.small_obj_conf if self.adaptive_conf else self.confidence,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],  # Person only
            device=self.device if self.device else None,
            verbose=False,
        )
        
        return self._parse_results(results)
    
    def _detect_sahi(self, frame: np.ndarray) -> List[Dict]:
        """SAHI tiled inference for small object detection."""
        result = get_sliced_prediction(
            frame,
            self.sahi_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=self.sahi_overlap,
            overlap_width_ratio=self.sahi_overlap,
        )
        
        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            cls_id = pred.category.id
            
            # Filter to person only (COCO class 0)
            if cls_id != self.PERSON_CLASS_ID:
                continue
            
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
            w, h = x2 - x1, y2 - y1
            area = w * h
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(pred.score.value),
                'label': 'person',
                'class_id': 0,
                'bbox_area': float(area),
                'is_small': max(w, h) < self.small_obj_threshold,
            })
        
        return detections
    
    def _parse_results(self, results) -> List[Dict]:
        """Parse ultralytics Results into standard detection dicts."""
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                w, h = x2 - x1, y2 - y1
                area = w * h
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'label': 'person',
                    'class_id': cls,
                    'bbox_area': area,
                    'is_small': max(w, h) < self.small_obj_threshold,
                })
        
        return detections
    
    def _adaptive_filter(self, detections: List[Dict]) -> List[Dict]:
        """Altitude-adaptive confidence filtering.
        
        Drone-specific: persons at high altitude appear very small.
        We use a lower confidence threshold for small detections
        and a higher one for large detections (to reduce false positives).
        """
        filtered = []
        for det in detections:
            if det['is_small']:
                # Small objects: accept lower confidence (they're inherently harder)
                if det['confidence'] >= self.small_obj_conf:
                    filtered.append(det)
            else:
                # Normal/large objects: standard threshold
                if det['confidence'] >= self.confidence:
                    filtered.append(det)
        return filtered
    
    def estimate_altitude_proxy(self, detections: List[Dict]) -> str:
        """Estimate relative altitude from average detection size.
        
        Returns: 'high' | 'medium' | 'low'
        Used for logging and adaptive parameter tuning.
        """
        if not detections:
            return 'unknown'
        
        avg_area = np.mean([d['bbox_area'] for d in detections])
        
        if avg_area < 800:      # very small persons → high altitude
            return 'high'
        elif avg_area < 3000:   # medium persons
            return 'medium'
        else:                   # large persons → low altitude  
            return 'low'
    
    @property
    def avg_fps(self) -> float:
        """Average FPS over recent frames."""
        if not self.inference_times:
            return 0.0
        recent = self.inference_times[-100:]
        avg_ms = sum(recent) / len(recent)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0
    
    @property
    def avg_ms(self) -> float:
        """Average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        recent = self.inference_times[-100:]
        return sum(recent) / len(recent)
    
    def get_stats(self) -> Dict:
        """Get detection statistics."""
        return {
            'model': str(self.model.model_name) if hasattr(self.model, 'model_name') else 'yolo11n',
            'imgsz': self.imgsz,
            'sahi': self.use_sahi,
            'adaptive_conf': self.adaptive_conf,
            'avg_ms': round(self.avg_ms, 1),
            'avg_fps': round(self.avg_fps, 1),
            'total_frames': self.frame_count,
        }
