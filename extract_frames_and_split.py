# save as extract_frames_and_split.py
import cv2
import os
import argparse
from pathlib import Path
import random
from tqdm import tqdm

def extract_frames(video_path, out_dir, every_n_frames=5, max_frames=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc=f"Extracting {video_path.name}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            fname = out_dir / f"{video_path.stem}_frame_{idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"Saved {saved} frames to {out_dir}")

def split_dataset(img_dir, train_dir, val_dir, val_ratio=0.15, seed=42):
    img_dir = Path(img_dir)
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    random.Random(seed).shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    val = imgs[:n_val]
    train = imgs[n_val:]
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    for p in train:
        os.rename(p, Path(train_dir) / p.name)
    for p in val:
        os.rename(p, Path(val_dir) / p.name)
    print(f"Moved {len(train)} train and {len(val)} val images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--out", type=str, default="data/raw_frames", help="Output dir for frames")
    parser.add_argument("--every", type=int, default=10, help="Save every n-th frame")
    parser.add_argument("--max", type=int, default=None, help="Max frames to save")
    parser.add_argument("--split", action="store_true", help="Split to train/val after extraction")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    video_path = Path(args.video)
    extract_frames(video_path, args.out, every_n_frames=args.every, max_frames=args.max)

    if args.split:
        # put train/val at data/images/train, data/images/val
        split_dataset(args.out, "data/images/train", "data/images/val", val_ratio=args.val_ratio)
