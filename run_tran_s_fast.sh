#uv run yolo task=detect mode=train model=runs/detect/train9/weights/best.pt data=/home/nssd/gled/vb/dataset-vb/backline/annotaned_ball_crop_yolov8_dataset_v2/data.yaml epochs=50 imgsz=640 plots=True batch=36 patience=50 # augment

uv run yolo task=detect mode=train model=models/default/yolo11s.pt data=/home/nssd/gled/vb/dataset-vb/backline/beack_crop/data.yaml epochs=50 imgsz=640 plots=True batch=16 augment # patience=50 # augment
