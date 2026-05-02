"""
MOT Metrics Evaluation — MOTA, IDF1, ID Switches
===================================================
Aerial Guardian | VisDrone MOT Pipeline

Evaluates tracking results against VisDrone ground truth annotations.
Computes standard MOT metrics:
  - MOTA (Multi-Object Tracking Accuracy)
  - IDF1 (ID F1 Score — identity preservation quality)
  - ID Switches (number of identity changes)
  - Mostly Tracked / Mostly Lost
  - FP / FN counts

Usage:
  # Evaluate a single sequence
  python evaluate.py --results output/mot_results/uav0000086_00000_v.txt --gt data/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt

  # Evaluate ALL sequences
  python evaluate.py --results-dir output/mot_results/ --gt-dir data/VisDrone2019-MOT-val/annotations/

  # Compare CMC ON vs OFF
  python evaluate.py --results-dir output/mot_cmc_on/ --gt-dir data/VisDrone2019-MOT-val/annotations/ --name "CMC ON"
  python evaluate.py --results-dir output/mot_cmc_off/ --gt-dir data/VisDrone2019-MOT-val/annotations/ --name "CMC OFF"
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from loguru import logger

try:
    import motmetrics as mm
    MOTMETRICS_AVAILABLE = True
except ImportError:
    MOTMETRICS_AVAILABLE = False
    logger.warning("motmetrics not installed. Run: pip install motmetrics")

# VisDrone person classes
PERSON_CLASSES = {1, 2}


def load_gt_annotations(gt_path: str, person_only: bool = True) -> Dict[int, List]:
    """Load VisDrone ground truth annotations.
    
    Args:
        gt_path: Path to annotation .txt file
        person_only: If True, only load person class annotations
        
    Returns:
        Dict mapping frame_id -> list of (track_id, x1, y1, w, h)
    """
    gt = defaultdict(list)
    
    with open(gt_path, 'r') as f:
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
            
            # Skip ignored regions
            if score == 0:
                continue
            
            # Filter to person only
            if person_only and category not in PERSON_CLASSES:
                continue
            
            gt[frame_id].append((track_id, x, y, w, h))
    
    return dict(gt)


def load_mot_results(result_path: str) -> Dict[int, List]:
    """Load tracking results in MOT format.
    
    MOT format: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1
    
    Returns:
        Dict mapping frame_id -> list of (track_id, x, y, w, h, conf)
    """
    results = defaultdict(list)
    
    with open(result_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            
            results[frame_id].append((track_id, x, y, w, h, conf))
    
    return dict(results)


def evaluate_sequence(
    gt: Dict[int, List],
    results: Dict[int, List],
    iou_threshold: float = 0.5,
) -> mm.MOTAccumulator:
    """Evaluate a single sequence using motmetrics.
    
    Args:
        gt: Ground truth dict from load_gt_annotations
        results: Tracking results dict from load_mot_results
        iou_threshold: IoU threshold for matching (standard: 0.5)
        
    Returns:
        MOTAccumulator with frame-by-frame matching results
    """
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Get all frame IDs from both GT and results
    all_frames = sorted(set(list(gt.keys()) + list(results.keys())))
    
    for frame_id in all_frames:
        gt_objects = gt.get(frame_id, [])
        pred_objects = results.get(frame_id, [])
        
        # Extract IDs and bboxes
        gt_ids = [obj[0] for obj in gt_objects]
        gt_bboxes = np.array([[obj[1], obj[2], obj[3], obj[4]] for obj in gt_objects])  # x, y, w, h
        
        pred_ids = [obj[0] for obj in pred_objects]
        pred_bboxes = np.array([[obj[1], obj[2], obj[3], obj[4]] for obj in pred_objects])  # x, y, w, h
        
        # Compute distance matrix (1 - IoU)
        if len(gt_bboxes) > 0 and len(pred_bboxes) > 0:
            distances = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=1 - iou_threshold)
        else:
            distances = np.empty((len(gt_bboxes), len(pred_bboxes)))
        
        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)
    
    return acc


def compute_metrics(accumulators: List, names: List[str]) -> None:
    """Compute and print MOT metrics summary.
    
    Args:
        accumulators: List of MOTAccumulator objects
        names: List of sequence names
    """
    mh = mm.metrics.create()
    
    summary = mh.compute_many(
        accumulators,
        names=names,
        metrics=[
            'mota',        # Multi-Object Tracking Accuracy
            'motp',        # Multi-Object Tracking Precision
            'idf1',        # ID F1 Score (identity preservation)
            'num_switches', # ID switches
            'num_false_positives',
            'num_misses',   # False negatives
            'num_objects',  # Total GT objects
            'num_predictions',
            'mostly_tracked',
            'mostly_lost',
            'num_fragmentations',
            'precision',
            'recall',
        ],
        generate_overall=True,
    )
    
    # Format and print
    formatters = mh.formatters
    formatters['mota'] = '{:.1%}'.format
    formatters['motp'] = '{:.3f}'.format
    formatters['idf1'] = '{:.1%}'.format
    formatters['precision'] = '{:.1%}'.format
    formatters['recall'] = '{:.1%}'.format
    
    print("\n" + "=" * 100)
    print("  MOT EVALUATION RESULTS")
    print("=" * 100)
    
    strsummary = mm.io.render_summary(
        summary,
        formatters=formatters,
        namemap={
            'mota': 'MOTA',
            'motp': 'MOTP',
            'idf1': 'IDF1',
            'num_switches': 'IDSw',
            'num_false_positives': 'FP',
            'num_misses': 'FN',
            'num_objects': 'GT',
            'num_predictions': 'Pred',
            'mostly_tracked': 'MT',
            'mostly_lost': 'ML',
            'num_fragmentations': 'FM',
            'precision': 'Prec',
            'recall': 'Rcll',
        },
    )
    
    print(strsummary)
    print("=" * 100)
    
    # Also print key metrics in a clean format
    overall = summary.loc['OVERALL']
    print(f"\n  Key Metrics:")
    print(f"    MOTA:   {overall['mota']:.1%}")
    print(f"    IDF1:   {overall['idf1']:.1%}")
    print(f"    IDSw:   {int(overall['num_switches'])}")
    print(f"    Prec:   {overall['precision']:.1%}")
    print(f"    Recall: {overall['recall']:.1%}")
    print(f"    MT/ML:  {int(overall['mostly_tracked'])} / {int(overall['mostly_lost'])}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MOT results against VisDrone ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Single sequence evaluation
    parser.add_argument('--results', default=None, help='Single MOT result file (.txt)')
    parser.add_argument('--gt', default=None, help='Single GT annotation file (.txt)')
    
    # Multi-sequence evaluation
    parser.add_argument('--results-dir', default=None, help='Directory with MOT result files')
    parser.add_argument('--gt-dir', default=None, help='Directory with GT annotation files')
    
    # Options
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IoU threshold for matching')
    parser.add_argument('--name', default='', help='Experiment name for display')
    
    args = parser.parse_args()
    
    if not MOTMETRICS_AVAILABLE:
        logger.error("motmetrics not installed. Run: pip install motmetrics")
        sys.exit(1)
    
    accumulators = []
    names = []
    
    if args.results and args.gt:
        # Single sequence
        logger.info(f"Evaluating: {args.results}")
        gt = load_gt_annotations(args.gt, person_only=True)
        results = load_mot_results(args.results)
        acc = evaluate_sequence(gt, results, args.iou_threshold)
        
        seq_name = Path(args.results).stem
        accumulators.append(acc)
        names.append(seq_name)
        
    elif args.results_dir and args.gt_dir:
        # Multiple sequences
        results_dir = Path(args.results_dir)
        gt_dir = Path(args.gt_dir)
        
        result_files = sorted(results_dir.glob("*.txt"))
        if not result_files:
            logger.error(f"No .txt files found in {results_dir}")
            sys.exit(1)
        
        for result_file in result_files:
            seq_name = result_file.stem
            gt_file = gt_dir / f"{seq_name}.txt"
            
            if not gt_file.exists():
                logger.warning(f"No GT file for {seq_name}, skipping")
                continue
            
            logger.info(f"Evaluating: {seq_name}")
            gt = load_gt_annotations(str(gt_file), person_only=True)
            results = load_mot_results(str(result_file))
            acc = evaluate_sequence(gt, results, args.iou_threshold)
            
            accumulators.append(acc)
            names.append(seq_name)
    
    else:
        parser.error("Provide either --results/--gt or --results-dir/--gt-dir")
    
    if not accumulators:
        logger.error("No sequences to evaluate")
        sys.exit(1)
    
    # Compute and display metrics
    experiment_name = args.name or "Aerial Guardian"
    print(f"\n  Experiment: {experiment_name}")
    summary = compute_metrics(accumulators, names)
    
    return summary


if __name__ == '__main__':
    main()
