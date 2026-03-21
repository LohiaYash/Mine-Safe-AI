import cv2
import numpy as np
import argparse

def detect_rocks_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # enhance edges
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Use adaptive threshold or Canny
    edges = cv2.Canny(blur, 50, 150)
    # dilate to close gaps
    kernel = np.ones((5,5), np.uint8)
    dil = cv2.dilate(edges, kernel, iterations=1)
    # find contours
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = gray.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # filter small blobs (tune)
            continue
        x,y,ww,hh = cv2.boundingRect(cnt)
        # filter extreme aspect ratios
        if ww/hh > 3 or hh/ww > 3: continue
        boxes.append((x,y,x+ww,y+hh))
    return boxes

def process_video(infile, outfile):
    cap = cv2.VideoCapture(infile)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(outfile, fourcc, fps, (w,h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detect_rocks_frame(frame)
        for (x1,y1,x2,y2) in boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        out.write(frame)
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", default="cv_out.mp4")
    args = parser.parse_args()
    process_video(args.infile, args.outfile)
    print("Done.")
