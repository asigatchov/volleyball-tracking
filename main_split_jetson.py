#!/usr/bin/python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import numpy as np
import cv2
from helper import create_video_writer
from collections import deque
def nearest(res,init):
    box = res.xyxy.cpu().numpy().astype('int')
    mean_c13 = np.mean(box[:, [0, 2]], axis=1)
    mean_c24 = np.mean(box[:, [1, 3]], axis=1)
    center = np.stack((mean_c13, mean_c24), axis=1).astype('int')
    distances = np.linalg.norm(center - init, axis=1).astype('int')
    return center[np.argmin(distances)],min(distances),np.argmin(distances)


file_model = 'model/yolo11_ball_640_crop_v202504092243.engine'
model = YOLO(file_model) # model name
#model.to('cuda')

video_file = "/apps/video/game_019.mp4" # file name

cap = cv2.VideoCapture(video_file) # file name
#cap = cv2.VideoCapture("video/game_p1_006.mp4") # file name

writer = create_video_writer(cap, "Output6.mp4")  # output file name

dq = deque(maxlen=15)
z=0
frame_num = 0
skip_spam = {}
while True:
    z += 1
    print(z)
    success, img = cap.read()
    if not success:
        break
 
    frame_num += 1

    height, width, _ = img.shape
    part_height = height // 2
    part_width = width // 3

    sub_images = [
        img[0:part_height, 0:part_width],  # Верхний левый
        img[0:part_height, part_width:2*part_width],  # Верхний средний
        img[0:part_height, 2*part_width:width],  # Верхний правый
        img[part_height:height, 0:part_width],  # Нижний левый
        img[part_height:height, part_width:2*part_width],  # Нижний средний
        img[part_height:height, 2*part_width:width],  # Нижний правый
    ]

    # Обработка частей батчем
    
    # Объединение результатов обратно в один кадр
    for idx, c_img in enumerate(sub_images):

        results = model(c_img, stream=True)
        for r in results:
            boxes = r.boxes
            y_offset = 0 if idx < 3 else part_height
            x_offset = (idx % 3) * part_width
            boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype('int') == 0]

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
                conf = box.conf[0].cpu().numpy()
            
                if conf > 0.7:
                    # Смещение координат обратно в общий кадр
                    x1 += x_offset
                    x2 += x_offset
                    y1 += y_offset
                    y2 += y_offset

                    cv2.circle(img, tuple((int((x1 + x2) / 2), int((y1 + y2) / 2))), 15, (255, 0, 0), 2)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    if skip_spam.get(center) is None:
                        skip_spam[center] = 1

                    skip_spam[center] += 1
                    print('coord:', center, skip_spam[center])        
                else:
                    pass
                    #if len(dq) > 1:
                    #    dq.pop()
            for i in range(1, len(dq)):
                if dq[i - 1] is None or dq[i] is None:
                    continue
                cv2.line(img, dq[i - 1], dq[i], (0, 0, 255), thickness=5)

    writer.write(img)
    

cap.release()
writer.release()
cv2.destroyAllWindows()
