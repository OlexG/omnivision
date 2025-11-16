from ultralytics import YOLO
model = YOLO("yolov8s.pt")
# Update data.yaml for our dataset
# data.yaml:
# Format:
# data.yaml
# path: /path/to/dataset
# train: images/train
# val: images/val
# test: images/test  # optional
# nc: 1
# names: [my_object]

model.train(data="data.yaml", imgsz=640, epochs=100, batch=-1, device='cpu', patience=20)
