# save as infer_video.py
from ultralytics import YOLO
import argparse
import cv2
from pathlib import Path
import numpy as np

def detect_with_ultralytics_weights(weights, source_video, out_dir="runs/detect_video", conf=0.25):
    model = YOLO(weights)
    # ultralytics will save an annotated video automatically if source is a video and save=True
    results = model.predict(source=source_video, save=True, conf=conf, save_txt=False, project=out_dir, name="rock_video")
    print("Saved results under:", out_dir)

def manual_frame_inference(weights, source_video, output_video, conf=0.25):
    # More control: read frames, run model per-frame, draw boxes via OpenCV
    model = YOLO(weights)
    cap = cv2.VideoCapture(source_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # model(frame) returns a Results object list
        results = model(frame, conf=conf)
        # results is list-like (one per batch element). We're passing single image
        r = results[0]
        # r.boxes.xyxy, r.boxes.conf, r.boxes.cls (they are tensors; convert to numpy)
        if r.boxes is not None and len(r.boxes) > 0:
            for box, confs, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                x1,y1,x2,y2 = box.astype(int)
                label = f"rock {confs:.2f}"
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        out.write(frame)
    cap.release()
    out.release()
    print("Wrote output:", output_video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--out", type=str, default="output_annotated.mp4")
    parser.add_argument("--mode", choices=["auto","manual"], default="auto")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.mode == "auto":
        detect_with_ultralytics_weights(args.weights, args.source, out_dir="runs/detect_video", conf=args.conf)
    else:
        manual_frame_inference(args.weights, args.source, args.out, conf=args.conf)
