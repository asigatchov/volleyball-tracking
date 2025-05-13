from ultralytics import YOLO

# model = YOLO('yolo11.yaml')  # или 'yolov8s-seg' для сегментации
#
last_model = "models/defaults/yolo11n.pt"
#last_model = 'models/asigatchov/yolo11n_gray3frame_ball_1024_e80_1k.pt'
model = YOLO(last_model)

data = "/home/projects/vb-soft/datasets/gray3frame/crop/data.yaml"
data = "/home/projects/vb-soft/datasets/home/crop/data.yaml"

# data = '/home/projects/www/vb-soft/datasets/hokkey/crop/data.yaml'
# В data.yaml
#   hsv_h: 0.0  # Отключить изменение оттенка (не нужно для grayscale)
#   hsv_s: 0.0
#   hsv_v: 0.7  # Изменение яркости
#   translate: 0.1
#   scale: 0.5
#   mosaic: 0.0  # Отключить мозаику, если кадры последовательны


model.train(
    data=data,
    imgsz=1024,
    epochs=100,
    batch=24,
    lr0=1e-4,
    augment=True,  # false не сработало?
    scale=0.2,
    box=12,  # усиленный вес для bbox loss
    cls=0.5,  # уменьшенный вес классификации
    hsv_h=0.0,  # 0.3
    hsv_s=0.0,  # 0.5
    hsv_v = 0.7 , # Изменение яркости
    degrees=5,
    flipud=0.5,
    mixup=0.2,
    translate = 0.1,
    cache=True,
    optimizer="AdamW",
    pretrained=True,
    rect=True,  # Прямоугольное обучение
    mosaic=0.0,  # Отключить мозаику, если кадры последовательны
    #    resume=True,
)

