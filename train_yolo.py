# save as train_yolo.py
from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data.yaml", help="data yaml path")
parser.add_argument("--model", type=str, default="yolov8n.pt", help="pretrained model to start from")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()

# Create and train model
model = YOLO(args.model)  # loads pretrained weights (nano)
model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
# after training, best weights are in runs/train/exp/weights/best.pt
