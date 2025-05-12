from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
        data="/home/projects/neuroweb/image-annotator/data/output/domain1_cls/",
        batch   = 64,
        translate = 0.0,
        scale = 0,
        flipud = 0.5,
        fliplr = 0.5,
        mosaic = 0.0,
        erasing = 0.0,
        hsv_s = 0.5,
        hsv_v = 0.3,
        epochs=200,
        augment = False,
        auto_augment = False,
        imgsz=224
        )
        