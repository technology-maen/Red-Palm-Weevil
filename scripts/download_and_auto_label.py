#!/usr/bin/env python3
"""
Download images referenced in `palma_dataset_1.0.csv` and create YOLO-format labels
that mark the whole image as class 0 (palm).

Usage:
  python scripts/download_and_auto_label.py --csv "palma_dataset_1.0.csv" --out dataset --max-images 0

Options:
  --csv         Path to CSV (default: palma_dataset_1.0.csv in repo)
  --out         Output dataset root (default: dataset)
  --max-images  Limit number of images to download (0 means all)
  --workers     Number of concurrent download workers (default: 8)
  --timeout     HTTP timeout seconds (default: 15)

Notes:
  - This script performs a best-effort download of URLs found in the Photos column of the CSV.
  - It auto-creates YOLO labels that cover the entire image: single line `0 0.5 0.5 1.0 1.0`.
  - After download it splits data into train/val/test (80/10/10) and creates `palm.yaml`.
"""

import argparse
import csv
import hashlib
import os
import pathlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit

import requests


def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def url_to_filename(url: str) -> str:
    # use sha1 of url + extension
    parsed = urlsplit(url)
    _, ext = os.path.splitext(parsed.path)
    if not ext:
        ext = '.jpg'
    h = hashlib.sha1(url.encode('utf-8')).hexdigest()
    return f"{h}{ext}"


def download_url(url: str, dest: pathlib.Path, timeout: int = 15) -> pathlib.Path | None:
    fname = url_to_filename(url)
    out = dest / fname
    if out.exists():
        return out
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(out, 'wb') as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
        return out
    except Exception as e:
        # remove partial file
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        return None


def read_photo_urls(csv_path: pathlib.Path) -> list[str]:
    urls = []
    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        if 'Photos' not in reader.fieldnames:
            print('CSV does not contain Photos column', file=sys.stderr)
            return []
        for row in reader:
            photos = row.get('Photos') or ''
            # photos often separated by ';'
            parts = [p.strip() for p in photos.split(';') if p.strip()]
            for p in parts:
                urls.append(p)
    # dedupe while preserving order
    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def write_full_image_label(img_path: pathlib.Path, labels_dir: pathlib.Path):
    # create label file with single bbox that covers the image: class 0, x=0.5 y=0.5 w=1.0 h=1.0
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_file = labels_dir / (img_path.stem + '.txt')
    with open(label_file, 'w', encoding='utf-8') as fh:
        fh.write('0 0.5 0.5 1.0 1.0\n')


def split_dataset(images_dir: pathlib.Path, labels_dir: pathlib.Path, out_root: pathlib.Path):
    # simple 80/10/10 split by file list
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    n = len(imgs)
    if n == 0:
        return
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        'train': imgs[:n_train],
        'val': imgs[n_train:n_train + n_val],
        'test': imgs[n_train + n_val:],
    }
    for split, files in splits.items():
        idir = out_root / 'images' / split
        ldir = out_root / 'labels' / split
        ensure_dir(idir)
        ensure_dir(ldir)
        for f in files:
            dest_img = idir / f.name
            if not dest_img.exists():
                shutil.copy2(f, dest_img)
            # copy label if exists
            lab = labels_dir / (f.stem + '.txt')
            if lab.exists():
                shutil.copy2(lab, ldir / lab.name)
            else:
                # create full-image label as fallback
                write_full_image_label(dest_img, ldir)


def write_yaml(out_root: pathlib.Path):
    yml = out_root / 'palm.yaml'
    content = f"train: '{(out_root / 'images' / 'train').absolute()}'\n"
    content += f"val: '{(out_root / 'images' / 'val').absolute()}'\n"
    content += f"test: '{(out_root / 'images' / 'test').absolute()}'\n"
    content += "\n"
    content += "nc: 1\n"
    content += "names: ['palm']\n"
    with open(yml, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print('Wrote', yml)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='palma_dataset_1.0.csv')
    p.add_argument('--out', default='dataset')
    p.add_argument('--max-images', type=int, default=0, help='0 means all')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--timeout', type=int, default=15)
    args = p.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        print('CSV not found:', csv_path)
        return

    urls = read_photo_urls(csv_path)
    print(f'Found {len(urls)} unique image URLs in CSV')
    if args.max_images > 0:
        urls = urls[: args.max_images]

    out_root = pathlib.Path(args.out)
    images_tmp = out_root / 'images_all'
    labels_tmp = out_root / 'labels_all'
    ensure_dir(images_tmp)
    ensure_dir(labels_tmp)

    # download with threads
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(download_url, u, images_tmp, args.timeout): u for u in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            res = None
            try:
                res = fut.result()
            except Exception as e:
                res = None
            if res is None:
                failures.append(url)

    print(f'Downloaded images to {images_tmp} (failures: {len(failures)})')
    if failures:
        print('Failed URLs sample:', failures[:5])

    # create full-image labels
    for img in images_tmp.iterdir():
        if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            write_full_image_label(img, labels_tmp)

    # split dataset
    split_dataset(images_tmp, labels_tmp, out_root)

    # write yaml
    write_yaml(pathlib.Path(args.out))

    print('Done. To train, run:')
    print("yolo detect train model=yolov8m.pt data='dataset/palm.yaml' epochs=200 imgsz=640 batch=8 name=palm_from_palma")


if __name__ == '__main__':
    main()
