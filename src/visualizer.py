"""
Track Visualizer — Premium Annotation Rendering
=================================================
Aerial Guardian | VisDrone MOT Pipeline

Renders:
  1. Bounding boxes with unique ID labels (color-coded per track)
  2. Trajectory tails with fade-out effect
  3. FPS counter and stats overlay
  4. Mini detection heatmap (optional)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# Visually distinct color palette (BGR format) — 20 unique colors
TRACK_COLORS = [
    (255, 64, 64),    # Coral blue
    (0, 255, 128),    # Spring green
    (255, 165, 0),    # Orange
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Amber
    (255, 255, 0),    # Light Cyan
    (64, 224, 208),   # Turquoise
    (255, 105, 180),  # Hot pink
    (50, 205, 50),    # Lime green
    (255, 215, 0),    # Gold
    (30, 144, 255),   # Dodger blue
    (255, 69, 0),     # Red orange
    (0, 206, 209),    # Turquoise
    (148, 103, 189),  # Muted purple
    (44, 160, 44),    # Green
    (214, 39, 40),    # Red
    (31, 119, 180),   # Blue
    (255, 127, 14),   # Orange
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Get a consistent color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


class TrackVisualizer:
    """Renders tracking results onto video frames.
    
    Features:
    - Color-coded bounding boxes with track IDs
    - Trajectory tail lines with fade-out
    - FPS and stats HUD overlay
    - Confidence labels
    """
    
    def __init__(
        self,
        bbox_thickness: int = 2,
        trail_length: int = 40,
        trail_thickness: int = 2,
        show_confidence: bool = True,
        show_trail: bool = True,
        show_hud: bool = True,
        font_scale: float = 0.6,
    ):
        self.bbox_thickness = bbox_thickness
        self.trail_length = trail_length
        self.trail_thickness = trail_thickness
        self.show_confidence = show_confidence
        self.show_trail = show_trail
        self.show_hud = show_hud
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
    
    def draw(
        self,
        frame: np.ndarray,
        tracks: list,
        fps: float = 0.0,
        frame_id: int = 0,
        det_count: int = 0,
        extra_stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """Draw all tracking annotations onto a frame.
        
        Args:
            frame: BGR image
            tracks: List of Track objects from DroneByteTracker
            fps: Current pipeline FPS
            frame_id: Current frame number
            det_count: Number of raw detections this frame
            extra_stats: Optional dict of extra stats to display
            
        Returns:
            Annotated BGR frame
        """
        annotated = frame.copy()
        
        # Draw trajectory tails FIRST (behind boxes)
        if self.show_trail:
            for track in tracks:
                self._draw_trail(annotated, track)
        
        # Draw bounding boxes and labels
        for track in tracks:
            self._draw_bbox(annotated, track)
        
        # Draw HUD overlay
        if self.show_hud:
            self._draw_hud(annotated, fps, frame_id, len(tracks), det_count, extra_stats)
        
        return annotated
    
    def _draw_bbox(self, frame: np.ndarray, track):
        """Draw bounding box with ID label."""
        color = get_track_color(track.track_id)
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # Build label
        label = f"ID:{track.track_id}"
        if self.show_confidence:
            label += f" {track.confidence:.2f}"
        
        # Label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, 1
        )
        
        # Position label above bbox
        label_y = max(y1 - 8, label_h + 4)
        
        cv2.rectangle(
            frame,
            (x1, label_y - label_h - 4),
            (x1 + label_w + 4, label_y + 4),
            color,
            -1,  # Filled
        )
        
        # Text (white on colored background)
        cv2.putText(
            frame, label,
            (x1 + 2, label_y),
            self.font, self.font_scale,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    
    def _draw_trail(self, frame: np.ndarray, track):
        """Draw trajectory tail with fade-out effect."""
        traj = track.trajectory
        if len(traj) < 2:
            return
        
        color = get_track_color(track.track_id)
        n_points = min(len(traj), self.trail_length)
        points = traj[-n_points:]
        
        for i in range(1, len(points)):
            # Fade effect: older points are more transparent
            alpha = i / len(points)  # 0 (oldest) → 1 (newest)
            thickness = max(1, int(self.trail_thickness * alpha))
            
            # Interpolate color towards background (fade)
            faded_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
            
            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            
            cv2.line(frame, pt1, pt2, faded_color, thickness, cv2.LINE_AA)
        
        # Draw a dot at the current position
        latest = (int(points[-1][0]), int(points[-1][1]))
        cv2.circle(frame, latest, 3, color, -1, cv2.LINE_AA)
    
    def _draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        frame_id: int,
        n_tracks: int,
        det_count: int,
        extra_stats: Optional[Dict] = None,
    ):
        """Draw heads-up display with stats."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        hud_h = 110  
        cv2.rectangle(overlay, (0, 0), (280, hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Stats text
        y_offset = 22
        line_height = 20
        
        # Title
        cv2.putText(frame, "AERIAL GUARDIAN", (8, y_offset),
                    self.font, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        y_offset += line_height
        
        # FPS (color-coded)
        fps_color = (0, 255, 0) if fps >= 25 else (0, 200, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (8, y_offset),
                    self.font, 0.5, fps_color, 1, cv2.LINE_AA)
        y_offset += line_height
        
        # Frame and track counts
        cv2.putText(frame, f"Frame: {frame_id} | Tracks: {n_tracks} | Dets: {det_count}",
                    (8, y_offset), self.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        y_offset += line_height
        
        # Extra stats
        if extra_stats:
            info = " | ".join(f"{k}: {v}" for k, v in extra_stats.items())
            cv2.putText(frame, info, (8, y_offset),
                        self.font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    
    def create_video_writer(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
    ) -> cv2.VideoWriter:
        """Create a video writer for output."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {output_path}")
        return writer
