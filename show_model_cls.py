from ultralytics import YOLO


files = ["yolo11s.pt",
         "yolo11n.pt",
         "models/Volley.ball.yolo11n.pt",
         "yolo11x.pt",
         ]
for f in files:

    model = YOLO(f) # model name
    print('model:',f, 'cls:', model.names)
    print("-"*20)