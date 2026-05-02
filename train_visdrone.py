"""
Fine-tune YOLO11n on VisDrone Pedestrian Class
================================================
Aerial Guardian | VisDrone MOT Pipeline

Converts VisDrone MOT annotations to YOLO detection format,
then fine-tunes YOLO11n on pedestrian class only.

Step 1: Convert VisDrone MOT annotations → YOLO format
Step 2: Create data.yaml for training
Step 3: Fine-tune with small-object-optimized hyperparameters

Usage:
  # Convert + train
  python train_visdrone.py --data-root data/VisDrone2019-MOT-val --epochs 50 --imgsz 1280

  # Convert only (then train manually)
  python train_visdrone.py --data-root data/VisDrone2019-MOT-val --convert-only
"""

import os
import sys
import glob
import shutil
import random
import argparse
import cv2
from pathlib import Path
from loguru import logger

# VisDrone pedestrian class IDs
PERSON_CLASSES = {1, 2}  # 1=pedestrian, 2=people


def convert_visdrone_to_yolo(data_root: str, output_dir: str, val_split: float = 0.2):
    """Convert VisDrone MOT annotations to YOLO detection format.
    
    VisDrone MOT format:
      <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <score>, <category>, ...
    
    YOLO format:
      <class_id> <x_center> <y_center> <width> <height>  (all normalized 0-1)
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # Find sequences
    seq_dir = data_root / "sequences"
    ann_dir = data_root / "annotations"
    
    if not seq_dir.exists():
        seq_dir = data_root
        ann_dir = data_root.parent / "annotations"
    
    seq_folders = sorted([d for d in seq_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    logger.info(f"Found {len(seq_folders)} sequences in {data_root}")
    
    # Collect all (image_path, annotations) pairs
    all_samples = []
    
    for seq_folder in seq_folders:
        ann_file = ann_dir / f"{seq_folder.name}.txt"
        
        if not ann_file.exists():
            logger.warning(f"No annotation file for {seq_folder.name}")
            continue
        
        # Read all images in sequence
        image_files = sorted(glob.glob(str(seq_folder / "*.jpg")))
        if not image_files:
            image_files = sorted(glob.glob(str(seq_folder / "*.png")))
        
        if not image_files:
            continue
        
        # Get image dimensions from first frame
        sample = cv2.imread(image_files[0])
        img_h, img_w = sample.shape[:2]
        
        # Parse annotations grouped by frame
        frame_annotations = {}
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                
                frame_id = int(parts[0])
                bb_left = int(parts[2])
                bb_top = int(parts[3])
                bb_width = int(parts[4])
                bb_height = int(parts[5])
                score = int(parts[6])
                category = int(parts[7])
                
                # Skip ignored regions and non-person classes
                if category not in PERSON_CLASSES:
                    continue
                if score == 0:  # ignored
                    continue
                
                # Convert to YOLO format (normalized center + size)
                x_center = (bb_left + bb_width / 2) / img_w
                y_center = (bb_top + bb_height / 2) / img_h
                w_norm = bb_width / img_w
                h_norm = bb_height / img_h
                
                # Clip to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w_norm = max(0, min(1, w_norm))
                h_norm = max(0, min(1, h_norm))
                
                if w_norm <= 0 or h_norm <= 0:
                    continue
                
                if frame_id not in frame_annotations:
                    frame_annotations[frame_id] = []
                
                # Class 0 = person (single class)
                frame_annotations[frame_id].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # Create samples — skip frames without persons (every 3rd to avoid redundancy)
        for i, img_path in enumerate(image_files):
            frame_id = i + 1  # 1-indexed
            
            # Sample every 3rd frame to reduce training redundancy
            if frame_id % 3 != 0:
                continue
            
            anns = frame_annotations.get(frame_id, [])
            if len(anns) == 0:
                # Keep ~10% of negative frames
                if random.random() > 0.1:
                    continue
            
            all_samples.append((img_path, anns, seq_folder.name, frame_id))
    
    logger.info(f"Total training samples: {len(all_samples)}")
    
    # Split into train/val
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - val_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    # Create output directory structure
    for split, samples in [('train', train_samples), ('val', val_samples)]:
        img_dir = output_dir / 'images' / split
        lbl_dir = output_dir / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path, anns, seq_name, frame_id in samples:
            # Copy image
            out_name = f"{seq_name}_{frame_id:07d}"
            ext = Path(img_path).suffix
            dst_img = img_dir / f"{out_name}{ext}"
            
            shutil.copy2(img_path, dst_img)
            
            # Write label file
            dst_lbl = lbl_dir / f"{out_name}.txt"
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(anns))
    
    logger.info(f"Train: {len(train_samples)} images | Val: {len(val_samples)} images")
    logger.info(f"Output: {output_dir}")
    
    # Create data.yaml
    data_yaml = output_dir / 'data.yaml'
    with open(data_yaml, 'w') as f:
        f.write(f"# VisDrone Pedestrian Detection Dataset\n")
        f.write(f"# Auto-generated by train_visdrone.py\n\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"# Single class: person (pedestrian + people merged)\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['person']\n")
    
    logger.info(f"Data config: {data_yaml}")
    return str(data_yaml)


def train(data_yaml: str, model: str = 'yolo11n.pt', epochs: int = 50, imgsz: int = 1280, 
          batch: int = 8, device: str = '', project: str = 'runs/visdrone'):
    """Fine-tune YOLO on VisDrone pedestrian data.
    
    Hyperparameters optimized for small aerial objects:
    - Higher resolution (1280) for better small-object detail
    - Mosaic augmentation for multi-scale learning
    - Copy-paste augmentation for small object diversity
    - Extended warmup for stable convergence
    """
    from ultralytics import YOLO
    
    logger.info(f"Fine-tuning {model} on VisDrone pedestrians")
    logger.info(f"  Epochs: {epochs} | ImgSz: {imgsz} | Batch: {batch}")
    
    yolo = YOLO(model)
    
    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=project,
        name='visdrone_person',
        
        # Small-object optimized hyperparameters
        lr0=0.01,
        lrf=0.01,            # Final LR factor
        warmup_epochs=5,
        cos_lr=True,          # Cosine LR schedule
        
        # Augmentation — crucial for small objects
        mosaic=1.0,           # Mosaic augmentation (essential for small objects)
        mixup=0.1,            # Light mixup
        copy_paste=0.3,       # Copy-paste for small object diversity
        degrees=10.0,         # Rotation
        translate=0.2,        # Translation
        scale=0.8,            # Scale jitter (important for altitude variation)
        shear=2.0,
        flipud=0.1,           # Vertical flip (drone can look down)
        fliplr=0.5,           # Horizontal flip
        
        # Close mosaic later for fine-grained learning
        close_mosaic=10,
        
        # Save best
        save=True,
        save_period=10,
        
        # Workers
        workers=4,
        
        verbose=True,
    )
    
    # Get best model path
    best_path = Path(project) / 'visdrone_person' / 'weights' / 'best.pt'
    if best_path.exists():
        logger.info(f"✅ Best model: {best_path}")
        logger.info(f"   Size: {best_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLO11n on VisDrone Pedestrians')
    parser.add_argument('--data-root', required=True, help='Path to VisDrone dataset root')
    parser.add_argument('--output', default='data/visdrone_yolo', help='Output YOLO-format dataset directory')
    parser.add_argument('--model', default='yolo11n.pt', help='Base YOLO model')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=1280, help='Training image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', default='', help='CUDA device')
    parser.add_argument('--convert-only', action='store_true', help='Only convert annotations, skip training')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Step 1: Convert annotations
    logger.info("Step 1: Converting VisDrone annotations to YOLO format...")
    data_yaml = convert_visdrone_to_yolo(args.data_root, args.output, args.val_split)
    
    if args.convert_only:
        logger.info("Conversion complete. Skipping training (--convert-only)")
        return
    
    # Step 2: Train
    logger.info("Step 2: Fine-tuning YOLO11n on VisDrone pedestrians...")
    train(data_yaml, args.model, args.epochs, args.imgsz, args.batch, args.device)
    
    logger.info("Done! Use the best.pt model in run_tracker.py:")
    logger.info("  python run_tracker.py --model runs/visdrone/visdrone_person/weights/best.pt --input ...")


if __name__ == '__main__':
    main()
