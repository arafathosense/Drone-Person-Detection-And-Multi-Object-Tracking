"""
Camera Motion Compensation (CMC) for Drone MOT
================================================
Aerial Guardian | VisDrone MOT Pipeline

Estimates frame-to-frame camera motion using ORB feature matching,
then warps previous track predictions to compensate for drone ego-motion.

This is the KEY differentiator for drone tracking vs ground-level tracking:
- Drone pan/tilt causes global image shift → Kalman predictions become invalid
- CMC warps previous bbox predictions into the current frame's coordinate system
- Result: ByteTrack sees "stabilized" predictions → fewer ID switches

Two modes:
  1. Affine (6 DOF) — for small rotations, translations, zoom (default, more stable)
  2. Homography (8 DOF) — for large perspective changes (optional)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from loguru import logger


class CameraMotionCompensator:
    """ORB-based camera motion compensation for drone tracking.
    
    Estimates affine/homography transform between consecutive frames
    using sparse ORB feature matching + RANSAC.
    
    Usage:
        cmc = CameraMotionCompensator()
        
        for frame in video:
            warp_matrix = cmc.estimate(frame)
            # Use warp_matrix to transform previous track bboxes
            warped_bboxes = cmc.warp_bboxes(prev_bboxes, warp_matrix, frame.shape)
    """
    
    def __init__(
        self,
        method: str = 'affine',     # 'affine' or 'homography'
        n_features: int = 1000,
        match_ratio: float = 0.75,  # Lowe's ratio test
        ransac_thresh: float = 5.0,
        min_matches: int = 20,
        downscale: float = 0.5,     # Process at half resolution for speed
    ):
        self.method = method
        self.n_features = n_features
        self.match_ratio = match_ratio
        self.ransac_thresh = ransac_thresh
        self.min_matches = min_matches
        self.downscale = downscale
        
        # ORB feature detector (fast binary descriptors — ideal for real-time)
        self.orb = cv2.ORB_create(nfeatures=n_features)
        
        # BF matcher for ORB (Hamming distance)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Previous frame state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_kp = None
        self.prev_des = None
        
        # Stats
        self.n_matches_history: List[int] = []
        self.warp_history: List[Optional[np.ndarray]] = []
    
    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate camera motion from previous frame to current frame.
        
        Args:
            frame: Current BGR frame
            
        Returns:
            Warp matrix (2x3 affine or 3x3 homography), or None if estimation failed.
            This matrix transforms points from PREVIOUS frame to CURRENT frame.
        """
        # Convert and optionally downscale for speed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.downscale != 1.0:
            h, w = gray.shape[:2]
            gray_small = cv2.resize(gray, (int(w * self.downscale), int(h * self.downscale)))
        else:
            gray_small = gray
        
        # Detect ORB features
        kp, des = self.orb.detectAndCompute(gray_small, None)
        
        warp_matrix = None
        
        if self.prev_gray is not None and self.prev_des is not None and des is not None:
            if len(kp) >= 10 and len(self.prev_kp) >= 10:
                warp_matrix = self._match_and_estimate(
                    self.prev_kp, self.prev_des,
                    kp, des,
                    gray_small.shape
                )
        
        # Update previous frame
        self.prev_gray = gray_small
        self.prev_kp = kp
        self.prev_des = des
        
        self.warp_history.append(warp_matrix)
        
        return warp_matrix
    
    def _match_and_estimate(
        self, kp1, des1, kp2, des2, shape
    ) -> Optional[np.ndarray]:
        """Match features and estimate warp transform."""
        
        # KNN match with ratio test (Lowe's)
        try:
            matches = self.matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            return None
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        self.n_matches_history.append(len(good_matches))
        
        if len(good_matches) < self.min_matches:
            return None
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Scale points back to original resolution
        if self.downscale != 1.0:
            scale = 1.0 / self.downscale
            src_pts *= scale
            dst_pts *= scale
        
        if self.method == 'affine':
            # Estimate affine transform (more stable for drone motion)
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
            )
            if M is None:
                return None
            n_inliers = np.sum(inliers) if inliers is not None else 0
            if n_inliers < self.min_matches * 0.4:
                return None
            return M  # 2x3 affine
        else:
            # Full homography (8 DOF)
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                self.ransac_thresh,
            )
            if H is None:
                return None
            n_inliers = np.sum(mask) if mask is not None else 0
            if n_inliers < self.min_matches * 0.4:
                return None
            return H  # 3x3 homography
    
    def warp_bboxes(
        self,
        bboxes: np.ndarray,
        warp_matrix: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Warp bounding boxes from previous frame coordinates to current frame.
        
        Args:
            bboxes: (N, 4) array of [x1, y1, x2, y2] in previous frame
            warp_matrix: 2x3 affine or 3x3 homography
            frame_shape: (H, W) of the current frame
            
        Returns:
            (N, 4) warped bounding boxes clipped to frame boundaries
        """
        if bboxes is None or len(bboxes) == 0:
            return bboxes
        
        h, w = frame_shape[:2]
        n = len(bboxes)
        
        # Extract corners of each bbox: top-left, top-right, bottom-right, bottom-left
        corners = np.zeros((n * 4, 2), dtype=np.float32)
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            corners[i*4 + 0] = [x1, y1]
            corners[i*4 + 1] = [x2, y1]
            corners[i*4 + 2] = [x2, y2]
            corners[i*4 + 3] = [x1, y2]
        
        corners = corners.reshape(-1, 1, 2)
        
        if warp_matrix.shape == (2, 3):
            # Affine transform
            warped = cv2.transform(corners, warp_matrix)
        else:
            # Perspective (homography)
            warped = cv2.perspectiveTransform(corners, warp_matrix)
        
        warped = warped.reshape(-1, 4, 2)
        
        # Reconstruct bboxes from warped corners
        warped_bboxes = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            x_coords = warped[i, :, 0]
            y_coords = warped[i, :, 1]
            warped_bboxes[i] = [
                max(0, np.min(x_coords)),
                max(0, np.min(y_coords)),
                min(w, np.max(x_coords)),
                min(h, np.max(y_coords)),
            ]
        
        return warped_bboxes
    
    def warp_points(
        self,
        points: np.ndarray,
        warp_matrix: np.ndarray,
    ) -> np.ndarray:
        """Warp 2D points from previous frame to current frame.
        
        Args:
            points: (N, 2) array of [x, y] points
            warp_matrix: 2x3 or 3x3 transform
            
        Returns:
            (N, 2) warped points
        """
        if points is None or len(points) == 0:
            return points
        
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        
        if warp_matrix.shape == (2, 3):
            warped = cv2.transform(pts, warp_matrix)
        else:
            warped = cv2.perspectiveTransform(pts, warp_matrix)
        
        return warped.reshape(-1, 2)
    
    @property
    def avg_matches(self) -> float:
        """Average number of feature matches."""
        if not self.n_matches_history:
            return 0.0
        recent = self.n_matches_history[-100:]
        return sum(recent) / len(recent)
    
    @property
    def success_rate(self) -> float:
        """Warp estimation success rate."""
        if not self.warp_history:
            return 0.0
        recent = self.warp_history[-100:]
        successes = sum(1 for w in recent if w is not None)
        return successes / len(recent)
    
    def get_stats(self) -> Dict:
        """Get CMC statistics."""
        return {
            'method': self.method,
            'avg_matches': round(self.avg_matches, 1),
            'success_rate': f"{self.success_rate*100:.1f}%",
            'total_frames': len(self.warp_history),
        }
    
    def motion_severity(self, warp_matrix: Optional[np.ndarray] = None) -> Dict:
        """Decompose warp matrix into interpretable motion components.
        
        This is the core of Ego-Motion-Adaptive Tracking (EMAT):
        Extract HOW MUCH the camera moved from the affine warp matrix,
        then use this as a feedback signal to dynamically adapt tracker
        parameters per frame.
        
        Standard ByteTrack/BoT-SORT use fixed parameters. EMAT makes them
        adaptive to the drone's ego-motion severity.
        
        Args:
            warp_matrix: 2x3 affine or 3x3 homography. If None, uses last estimated.
            
        Returns:
            Dict with motion components:
            - translation_px: Camera translation in pixels
            - rotation_deg: Camera rotation in degrees
            - scale_change: Scale factor (1.0 = no change)
            - severity: Normalized 0-1 motion severity score
        """
        if warp_matrix is None:
            warp_matrix = self.warp_history[-1] if self.warp_history else None
        
        if warp_matrix is None:
            return {
                'translation_px': 0.0,
                'rotation_deg': 0.0,
                'scale_change': 1.0,
                'severity': 0.0,
            }
        
        # For 3x3 homography, extract the top 2x3 affine part
        if warp_matrix.shape == (3, 3):
            M = warp_matrix[:2, :]
        else:
            M = warp_matrix
        
        # Decompose 2x3 affine: [[a, b, tx], [c, d, ty]]
        a, b, tx = M[0]
        c, d, ty = M[1]
        
        # Translation (in pixels)
        translation_px = np.sqrt(tx**2 + ty**2)
        
        # Rotation (from affine components)
        rotation_rad = np.arctan2(c, a)
        rotation_deg = np.degrees(rotation_rad)
        
        # Scale change
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)
        scale_change = (scale_x + scale_y) / 2.0
        
        # Severity score: normalized combination of motion components
        # Translation > 20px is significant, rotation > 2° is significant
        trans_severity = min(1.0, translation_px / 50.0)
        rot_severity = min(1.0, abs(rotation_deg) / 5.0)
        scale_severity = min(1.0, abs(scale_change - 1.0) / 0.1)
        
        severity = max(trans_severity, rot_severity, scale_severity)
        
        return {
            'translation_px': round(float(translation_px), 2),
            'rotation_deg': round(float(rotation_deg), 3),
            'scale_change': round(float(scale_change), 4),
            'severity': round(float(severity), 3),
        }
    
    def reset(self):
        """Reset CMC state (e.g. at sequence boundary)."""
        self.prev_gray = None
        self.prev_kp = None
        self.prev_des = None
        self.n_matches_history.clear()
        self.warp_history.clear()

