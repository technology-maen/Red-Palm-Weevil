"""
YOLO-based palm detector and counter

Usage (example):
  python test.py --model runs/detect/palm_yolov8n/weights/best.pt --source images/ --outdir outputs/ --class-name palm

This script uses Ultralytics YOLO (v8) Python API to run inference on images or a folder
and counts detections that match the requested class name (default: 'palm'). It saves
annotated images with bounding boxes and writes per-image counts to stdout.

Requirements:
  pip install -U ultralytics opencv-python-headless

If you don't have a trained palm model yet, train one with the `ultralytics` CLI or
provide a model path produced after training (e.g. runs/detect/.../weights/best.pt).
"""

import os
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("ultralytics is required. Install with: pip install ultralytics") from e

import cv2
import math
import numpy as np
from typing import Optional
try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def annotate_and_count(model, img_path: str, out_path: str, conf: float, iou: float, class_name: str | None,
                       meters_per_pixel: float | None = None, focal_px: float | None = None, distance_m: float | None = None,
                       depth_model: object | None = None, depth_transform: object | None = None, device: str = 'cpu'):
    """Run YOLO on a single image, annotate it and return the count of matching detections and heights.

    Returns:
      count (int), heights_px (list of int), heights_m (list of float or None)
    """
    results = model.predict(source=img_path, conf=conf, iou=iou, imgsz=640, verbose=False)
    if not results:
        return 0, [], []
    r = results[0]

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    count = 0
    heights_px: list[int] = []
    heights_m: list[float | None] = []
    names = getattr(model, 'names', {}) or {}

    # If depth model provided, compute depth map (MiDaS returns inverse-depth; we keep raw output as relative depth)
    depth_map = None
    if depth_model is not None and depth_transform is not None:
        try:
            import torch
            img_bgr = img
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            input_batch = depth_transform(Image.fromarray(img_rgb)).to(device)
            with torch.no_grad():
                prediction = depth_model(input_batch.unsqueeze(0))
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
                ).squeeze().cpu().numpy()
            depth_map = prediction
        except Exception:
            depth_map = None

    # Iterate over detected boxes
    for box in r.boxes:
        # Extract coordinates and metadata in a robust way
        try:
            coords = box.xyxy[0].tolist()
        except Exception:
            coords = [int(x) for x in box.xyxy[0]]

        x1, y1, x2, y2 = map(int, coords)
        conf_score = float(box.conf[0])
        cls_idx = int(box.cls[0])
        cls_name = names.get(cls_idx, str(cls_idx))

        # If a class_name filter is set, only count those that match (case-insensitive)
        if class_name is None or class_name.lower() in str(cls_name).lower():
            count += 1
            h_px = max(0, y2 - y1)
            heights_px.append(h_px)

            # Try to compute metric height if possible
            height_m = None
            depth_top = None
            depth_bottom = None
            # compute depths if depth_map available
            if depth_map is not None:
                import numpy as np
                # sample small regions near top and bottom for robust depth
                h = max(1, int((y2 - y1) * 0.05))
                w = max(1, int((x2 - x1) * 0.2))
                yt0 = max(0, y1)
                yt1 = min(img.shape[0], y1 + h)
                xb0 = max(0, x1)
                xb1 = min(img.shape[1], x1 + w)
                yb0 = max(0, y2 - h)
                yb1 = min(img.shape[0], y2)
                try:
                    depth_top = float(np.median(depth_map[yt0:yt1, xb0:xb1]))
                    depth_bottom = float(np.median(depth_map[yb0:yb1, xb0:xb1]))
                except Exception:
                    depth_top = None
                    depth_bottom = None
            if meters_per_pixel is not None:
                height_m = h_px * meters_per_pixel
            elif focal_px is not None and distance_m is not None:
                # pinhole camera: real_height ~= pixel_height * (distance / focal_length_pixels)
                try:
                    height_m = h_px * (distance_m / float(focal_px))
                except Exception:
                    height_m = None

            # If we have depth values (relative) and user also provided a known distance for scaling,
            # we can attempt a more informed estimate: scale the relative depth to the provided distance
            # and use pinhole relation per-detection. This is optional and requires --distance and --focal-px.
            if height_m is None and depth_top is not None and depth_bottom is not None and focal_px is not None and distance_m is not None:
                try:
                    # use mid-point depth as relative Z; compute scale factor from midas units to meters
                    z_rel = max(1e-6, (depth_top + depth_bottom) / 2.0)
                    # Note: MiDaS outputs relative depth. We scale it so that the median depth in image equals provided distance.
                    # This is a heuristic: better to provide a reference object for exact scaling.
                    z_m = distance_m * (z_rel / (z_rel + 0.0))
                    height_m = h_px * (z_m / float(focal_px))
                except Exception:
                    height_m = None

            heights_m.append(height_m)

            # Build label
            if height_m is None:
                label = f"{cls_name} {conf_score:.2f} H:{h_px}px"
            else:
                label = f"{cls_name} {conf_score:.2f} H:{h_px}px {height_m:.2f}m"
            if depth_top is not None and depth_bottom is not None:
                label += f" Dtop:{depth_top:.2f} Dbot:{depth_bottom:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Put total count text
    cv2.putText(img, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Save annotated image
    cv2.imwrite(out_path, img)
    return count, heights_px, heights_m


def iter_images(source: Path):
    if source.is_dir():
        for p in sorted(source.iterdir()):
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                yield p
    elif source.is_file():
        yield source
    else:
        raise FileNotFoundError(f"Source not found: {source}")


def get_focal_px_from_exif(image_path: Path) -> Optional[float]:
    """Try to extract focal length (35mm equivalent) from EXIF and convert to pixels.

    Returns focal length in pixels if possible, otherwise None.
    Heuristic: use FocalLengthIn35mmFilm (if present) and assume 36mm film width.
    """
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(str(image_path))
        exif = img._getexif()
        if not exif:
            return None
        focal_35 = None
        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, tag_id)
            if name == 'FocalLengthIn35mmFilm':
                try:
                    focal_35 = float(value)
                except Exception:
                    try:
                        # sometimes it's a tuple
                        focal_35 = float(value[0]) / float(value[1])
                    except Exception:
                        focal_35 = None
                break

        if focal_35 is not None:
            img_w = img.width
            # assume 36mm film width
            focal_px = (focal_35 / 36.0) * float(img_w)
            return float(focal_px)
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description='YOLO palm counter')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model or pretrained name')
    parser.add_argument('--source', type=str, default='images/', help='Image or folder to run inference on')
    parser.add_argument('--outdir', type=str, default='outputs/', help='Directory to save annotated images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--class-name', type=str, default='palm', help='Class name to count (case-insensitive). Use "None" to count all detections')
    parser.add_argument('--meters-per-pixel', type=float, default=None, help='Approximate meters per pixel scale to convert pixel heights to meters')
    parser.add_argument('--ref-pixel', type=float, default=None, help='Pixel height of a reference object in the image (used with --ref-meter to compute scale)')
    parser.add_argument('--ref-meter', type=float, default=None, help='Real-world height (meters) of the reference object (used with --ref-pixel)')
    parser.add_argument('--focal-px', type=float, default=None, help='Camera focal length in pixels (used with --distance to estimate heights)')
    parser.add_argument('--distance', type=float, default=None, help='Estimated distance (meters) to the trees (used with --focal-px)')
    parser.add_argument('--save-csv', action='store_true', help='Save per-image detection counts and heights to results.csv in outdir')
    parser.add_argument('--depth-mode', choices=['none', 'midas'], default='none', help='Optional depth estimation mode (midas for monocular depth)')
    args = parser.parse_args()

    # Interpret 'None' string as no filter
    class_name = None if str(args.class_name).lower() in ('none', 'null', '0', '') else args.class_name

    src = Path(args.source)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(args.model)

    # Prepare depth model if requested
    depth_model = None
    depth_transform = None
    device = 'cpu'
    if args.depth_mode == 'midas':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            depth_transform = midas_transforms.small_transform
            depth_model.to(device).eval()
            print(f"Loaded MiDaS model on {device}")
        except Exception as e:
            print(f"Warning: could not load MiDaS depth model: {e}. Continuing without depth.")
            depth_model = None
            depth_transform = None

    total = 0
    files_processed = 0
    csv_rows = []

    for img_path in iter_images(src):
        out_path = outdir / img_path.name
        try:
            # Determine effective meters_per_pixel based on user inputs
            meters_per_pixel = None
            if args.ref_pixel is not None and args.ref_meter is not None:
                # Use reference object to compute scale
                if args.ref_pixel > 0:
                    meters_per_pixel = float(args.ref_meter) / float(args.ref_pixel)
            elif args.meters_per_pixel is not None:
                meters_per_pixel = args.meters_per_pixel

            # Determine focal_px: prefer CLI, otherwise try EXIF on first image
            focal_px = args.focal_px
            if focal_px is None:
                try:
                    fx = get_focal_px_from_exif(img_path)
                    if fx is not None:
                        focal_px = fx
                        print(f"Using focal_px from EXIF for {img_path.name}: {focal_px:.1f} px (heuristic)")
                except Exception:
                    focal_px = None

            count, heights_px, heights_m = annotate_and_count(
                model, str(img_path), str(out_path), args.conf, args.iou, class_name,
                meters_per_pixel=meters_per_pixel, focal_px=focal_px, distance_m=args.distance,
                depth_model=depth_model, depth_transform=depth_transform, device=device,
            )
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

        # Format per-image output and collect CSV rows
        if heights_px:
            heights_px_str = ",".join(str(h) for h in heights_px)
            heights_m_str = ",".join((f"{hm:.2f}" if hm is not None else "N/A") for hm in heights_m)
            avg_px = sum(heights_px) / len(heights_px)
            valid_m = [hm for hm in heights_m if hm is not None]
            avg_m = (sum(valid_m) / len(valid_m)) if valid_m else None
            msg = f"{img_path.name}: {count} heights_px=[{heights_px_str}] heights_m=[{heights_m_str}] avg_px={avg_px:.1f}"
            if avg_m is not None:
                msg += f" avg_m={avg_m:.2f}m"
            print(msg)
            csv_rows.append((img_path.name, count, heights_px_str, heights_m_str, f"{avg_px:.1f}", f"{avg_m:.2f}" if avg_m is not None else "N/A"))
        else:
            print(f"{img_path.name}: {count}")
            csv_rows.append((img_path.name, count, "", "", "0", "N/A"))

        total += count
        files_processed += 1

    print("""
------------ Summary ------------
""")
    print(f"Images processed: {files_processed}")
    print(f"Total palms detected: {total}")

    # Save CSV if requested
    if args.save_csv:
        import csv
        csv_path = outdir / 'results.csv'
        with open(csv_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['image', 'count', 'heights_px', 'heights_m', 'avg_px', 'avg_m'])
            for row in csv_rows:
                writer.writerow(row)
        print(f"Saved CSV results to: {csv_path}")


if __name__ == '__main__':
    main()
