#uv run yolo task=detect mode=train model=runs/detect/train9/weights/best.pt data=/home/nssd/gled/vb/dataset-vb/backline/annotaned_ball_crop_yolov8_dataset_v2/data.yaml epochs=50 imgsz=640 plots=True batch=36 patience=50 # augment

#uv run yolo task=detect mode=train model=models/default/yolo11n.pt data=/home/nssd/gled/vb/dataset-vb/backline/join_ds/data.yaml epochs=150 imgsz=640 plots=True batch=25 augment  patience=30


uv run yolo  mode=train model=/home/projects/www/vb-soft/volleyball-tracking/runs/detect/train29/weights/best.pt  data=/home/nssd/gled/vb/dataset-vb/sideline/orel_pobeda/crop/data.yaml augment epochs=100 imgsz=640 plots=True batch=25 patience=30
