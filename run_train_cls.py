from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
        data="/home/nssd/gled/vb/dataset-vb/backline/beach_cls/",
        epochs=200, imgsz=64
        )
