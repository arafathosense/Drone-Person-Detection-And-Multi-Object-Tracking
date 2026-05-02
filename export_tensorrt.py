"""
Export YOLO to TensorRT for Jetson Orin Nano
==============================================
Aerial Guardian | VisDrone MOT Pipeline

Exports fine-tuned YOLO model for edge deployment:
  1. PyTorch → ONNX (platform-independent)
  2. ONNX → TensorRT FP16 (build on target Jetson)

Usage:
  # Export to ONNX (run on PC)
  python export_tensorrt.py --model runs/visdrone/visdrone_person/weights/best.pt --format onnx

  # Export to TensorRT FP16 (run on Jetson)
  python export_tensorrt.py --model best.pt --format engine --imgsz 1280

  # Export with INT8 quantization (run on Jetson with calibration data)
  python export_tensorrt.py --model best.pt --format engine --int8 --data data/visdrone_yolo/data.yaml
"""

import os
import sys
import argparse
from pathlib import Path
from loguru import logger


def export_model(
    model_path: str,
    format: str = 'onnx',
    imgsz: int = 1280,
    half: bool = True,
    int8: bool = False,
    data: str = None,
    device: str = '',
    simplify: bool = True,
    workspace: int = 4,
):
    """Export YOLO model to the specified format."""
    from ultralytics import YOLO
    
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    model_size_mb = os.path.getsize(model_path) / 1024 / 1024
    logger.info(f"  PyTorch model size: {model_size_mb:.1f} MB")
    
    logger.info(f"Exporting to {format}...")
    logger.info(f"  ImgSz: {imgsz} | Half: {half} | INT8: {int8}")
    
    export_args = dict(
        format=format,
        imgsz=imgsz,
        simplify=simplify,
    )
    
    if format == 'onnx':
        export_args['half'] = False  # ONNX doesn't support half directly
        export_args['dynamic'] = False
        export_args['opset'] = 17
    elif format == 'engine':
        export_args['half'] = half
        export_args['int8'] = int8
        export_args['workspace'] = workspace
        if int8 and data:
            export_args['data'] = data
    
    if device:
        export_args['device'] = device
    
    result = model.export(**export_args)
    
    # Report sizes
    if result:
        export_path = Path(result)
        if export_path.exists():
            export_size = export_path.stat().st_size / 1024 / 1024
            logger.info(f"\n✅ Export complete!")
            logger.info(f"  Format: {format}")
            logger.info(f"  Path: {export_path}")
            logger.info(f"  Size: {export_size:.1f} MB")
            logger.info(f"  Size check: {'✅ PASS' if export_size < 300 else '❌ EXCEEDS 300MB'} (limit: 300MB)")
            
            if format == 'onnx':
                logger.info(f"\nTo build TensorRT engine on Jetson:")
                logger.info(f"  scp {export_path} yash@<jetson_ip>:~/visdrone_mot/")
                logger.info(f"  # On Jetson:")
                logger.info(f"  python export_tensorrt.py --model {export_path.name} --format engine --imgsz {imgsz}")
    
    return result


def benchmark_model(model_path: str, imgsz: int = 1280, n_runs: int = 50, device: str = ''):
    """Benchmark model inference speed."""
    import time
    import numpy as np
    from ultralytics import YOLO
    
    logger.info(f"Benchmarking: {model_path}")
    model = YOLO(model_path)
    
    # Warmup
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, imgsz=imgsz, verbose=False, device=device if device else None)
    
    # Benchmark
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, verbose=False, device=device if device else None)
        times.append((time.perf_counter() - t0) * 1000)
    
    avg_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / avg_ms
    
    logger.info(f"\n📊 Benchmark Results ({n_runs} runs):")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Input: {imgsz}x{imgsz}")
    logger.info(f"  Latency: {avg_ms:.1f} ± {std_ms:.1f} ms")
    logger.info(f"  FPS: {fps:.1f}")
    logger.info(f"  Min: {min(times):.1f}ms | Max: {max(times):.1f}ms")
    
    return {'avg_ms': avg_ms, 'fps': fps, 'std_ms': std_ms}


def main():
    parser = argparse.ArgumentParser(description='Export YOLO to TensorRT for Jetson')
    parser.add_argument('--model', required=True, help='YOLO model path (.pt)')
    parser.add_argument('--format', default='onnx', choices=['onnx', 'engine', 'torchscript'], help='Export format')
    parser.add_argument('--imgsz', type=int, default=1280, help='Input resolution')
    parser.add_argument('--no-half', action='store_true', help='Disable FP16')
    parser.add_argument('--int8', action='store_true', help='INT8 quantization (TRT only)')
    parser.add_argument('--data', default=None, help='Calibration data for INT8')
    parser.add_argument('--device', default='', help='CUDA device')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark after export')
    parser.add_argument('--n-runs', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--workspace', type=int, default=4, help='TRT workspace size (GB)')
    
    args = parser.parse_args()
    
    # Export
    result = export_model(
        args.model, args.format, args.imgsz,
        half=not args.no_half,
        int8=args.int8,
        data=args.data,
        device=args.device,
        workspace=args.workspace,
    )
    
    # Optional benchmark
    if args.benchmark and result:
        benchmark_model(result, args.imgsz, args.n_runs, args.device)


if __name__ == '__main__':
    main()
