"""
VisDrone MOT Dataset Loader
============================
Aerial Guardian | VisDrone MOT Pipeline

Loads VisDrone2019-MOT validation sequences.
Supports:
  - Image sequence folders (standard VisDrone format)
  - Video files (.mp4, .avi)
  - Ground truth annotation loading

VisDrone MOT folder structure:
  VisDrone2019-MOT-val/
  ├── sequences/
  │   ├── uav0000013_00000_v/
  │   │   ├── 0000001.jpg
  │   │   ├── 0000002.jpg
  │   │   └── ...
  │   ├── uav0000013_01392_v/
  │   └── ...
  └── annotations/
      ├── uav0000013_00000_v.txt
      └── ...

Annotation format per line:
  <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <score>, <category>, <truncation>, <occlusion>

VisDrone categories:
  0: ignored, 1: pedestrian, 2: people, 3: bicycle, 4: car,
  5: van, 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus, 10: motor, 11: others
"""

import os
import glob
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Iterator
from pathlib import Path
from loguru import logger


# VisDrone class mapping
VISDRONE_CLASSES = {
    0: 'ignored',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others',
}

# Person-related classes
PERSON_CLASSES = {1, 2}  # pedestrian + people


class VisDroneSequence:
    """Single VisDrone MOT sequence."""
    
    def __init__(self, seq_path: str, annotation_path: Optional[str] = None):
        self.seq_path = Path(seq_path)
        self.name = self.seq_path.name
        
        # Find all frames
        self.frame_paths = sorted(glob.glob(str(self.seq_path / "*.jpg")))
        if not self.frame_paths:
            # Try png
            self.frame_paths = sorted(glob.glob(str(self.seq_path / "*.png")))
        
        self.n_frames = len(self.frame_paths)
        
        # Load first frame to get dimensions
        if self.n_frames > 0:
            sample = cv2.imread(self.frame_paths[0])
            self.height, self.width = sample.shape[:2]
        else:
            self.height, self.width = 0, 0
        
        # Load annotations if available
        self.annotations = {}  # frame_id -> list of annotations
        if annotation_path and os.path.exists(annotation_path):
            self._load_annotations(annotation_path)
        
        logger.info(f"Sequence '{self.name}': {self.n_frames} frames, "
                    f"{self.width}x{self.height}, "
                    f"{'with' if self.annotations else 'no'} annotations")
    
    def _load_annotations(self, ann_path: str):
        """Load VisDrone MOT annotations."""
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x = int(parts[2])
                y = int(parts[3])
                w = int(parts[4])
                h = int(parts[5])
                score = int(parts[6])
                category = int(parts[7])
                truncation = int(parts[8]) if len(parts) > 8 else 0
                occlusion = int(parts[9]) if len(parts) > 9 else 0
                
                if frame_id not in self.annotations:
                    self.annotations[frame_id] = []
                
                self.annotations[frame_id].append({
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'bbox': [x, y, x + w, y + h],  # Convert to [x1, y1, x2, y2]
                    'bbox_ltwh': [x, y, w, h],      # Keep original format too
                    'score': score,
                    'category': category,
                    'category_name': VISDRONE_CLASSES.get(category, 'unknown'),
                    'truncation': truncation,
                    'occlusion': occlusion,
                    'is_person': category in PERSON_CLASSES,
                })
    
    def get_person_annotations(self, frame_id: int) -> List[Dict]:
        """Get person-only annotations for a specific frame."""
        if frame_id not in self.annotations:
            return []
        return [a for a in self.annotations[frame_id] if a['is_person']]
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over frames. Yields (frame_id, frame_bgr)."""
        for i, path in enumerate(self.frame_paths):
            frame = cv2.imread(path)
            if frame is not None:
                yield i + 1, frame  # VisDrone uses 1-indexed frames
    
    def __len__(self) -> int:
        return self.n_frames
    
    def get_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get a specific frame by ID (1-indexed)."""
        idx = frame_id - 1
        if 0 <= idx < len(self.frame_paths):
            return cv2.imread(self.frame_paths[idx])
        return None


class VisDroneLoader:
    """Load all sequences from a VisDrone MOT dataset directory."""
    
    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root)
        self.sequences: List[VisDroneSequence] = []
        
        # Try to find sequences
        seq_dir = self.root / "sequences"
        ann_dir = self.root / "annotations"
        
        if not seq_dir.exists():
            # Maybe the root IS the sequences folder
            seq_dir = self.root
            ann_dir = self.root.parent / "annotations"
        
        if seq_dir.exists():
            seq_folders = sorted([
                d for d in seq_dir.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ])
            
            for seq_folder in seq_folders:
                ann_file = ann_dir / f"{seq_folder.name}.txt"
                ann_path = str(ann_file) if ann_file.exists() else None
                
                seq = VisDroneSequence(str(seq_folder), ann_path)
                if seq.n_frames > 0:
                    self.sequences.append(seq)
        
        logger.info(f"VisDrone dataset: {len(self.sequences)} sequences loaded from {dataset_root}")
    
    def __iter__(self) -> Iterator[VisDroneSequence]:
        return iter(self.sequences)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx) -> VisDroneSequence:
        return self.sequences[idx]
    
    @property
    def total_frames(self) -> int:
        return sum(len(s) for s in self.sequences)


class VideoLoader:
    """Load frames from a video file (alternative to VisDrone sequences)."""
    
    def __init__(self, video_path: str):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video: {video_path} | {self.n_frames} frames, "
                    f"{self.width}x{self.height} @ {self.fps:.1f} FPS")
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame
    
    def __len__(self) -> int:
        return self.n_frames
    
    def release(self):
        self.cap.release()
