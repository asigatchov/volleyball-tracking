from ultralytics import YOLO

#model = YOLO('yolo11.yaml')  # или 'yolov8s-seg' для сегментации
#
#model = YOLO('models/custom/last.pt')  # или 'yolov8s-seg' для сегментации
#
#model.train(
#    data='/home/nssd/gled/vb/dataset-vb/backline/join_ds/data.yaml',
#    imgsz=640,
#    epochs=140,
#    batch=28,
#    lr0=1e-4,
#    augment=True,
#    hsv_h=0.3,
#    hsv_s=0.5,
#    degrees=10,
#    flipud=0.5,
#    mixup=0.2,
#    cache=True,
#    optimizer='AdamW',
#    pretrained=True,
#   # resume=True
#)
#
#model = YOLO('yolo11.yaml').load('models/custom/last.pt')
model = YOLO('models/custom/last.pt')

# Конфигурация обучения
results = model.train(
    data='/home/nssd/gled/vb/dataset-vb/backline/join_ds/data.yaml',
    imgsz=640,  # увеличенное разрешение
    epochs=200,
    batch=28,
    optimizer='AdamW',
    lr0=1e-4,
    warmup_epochs=3,
    box=7.5,  # усиленный вес для bbox loss
    cls=0.5,  # уменьшенный вес классификации
    hsv_h=0.2,  # увеличенная аугментация цвета
    hsv_s=0.8,
    hsv_v=0.8,
    translate=0.2,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.2,
    anchor_t=3.0  # увеличенный порог для анкоров
    #close_mosaic=10
)
