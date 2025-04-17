from ultralytics import YOLO
import numpy as np
import cv2
from helper import create_video_writer
from collections import deque

import argparse

# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")
parser.add_argument("video_file", type=str, help="Path to the video file")
args = parser.parse_args()

video_file = (
    args.video_file
)  # Получение пути к видеофайлу из аргументов командной строки


def nearest(res,init):
    box = res.xyxy.cpu().numpy().astype('int')
    mean_c13 = np.mean(box[:, [0, 2]], axis=1)
    mean_c24 = np.mean(box[:, [1, 3]], axis=1)
    center = np.stack((mean_c13, mean_c24), axis=1).astype('int')
    distances = np.linalg.norm(center - init, axis=1).astype('int')
    return center[np.argmin(distances)],min(distances),np.argmin(distances)
# model = YOLO("yolo-default/yolo11s.pt") # model name


file_model = "runs/detect/train5/weights/best.pt"

file_model = "runs/detect/train7/weights/best.pt"
model = YOLO(file_model) # model name
model.to('cuda')
мяч = [0,0]

# cap = cv2.VideoCapture("video/p4/game_005.mp4") # file name
# cap = cv2.VideoCapture("video/man_pro.mp4") # file name
# cap = cv2.VideoCapture("video/game_019.mp4") # file name

#video_file = 'video/man_pro.mp4'
#video_file = 'video/p4/game_005.mp4'

cap = cv2.VideoCapture(video_file) # file name

writer = create_video_writer(cap, "Output6.mp4")  # output file name

dq = deque(maxlen=10)
z=0
while True:
    print(z)
    z +=1
    success, img = cap.read()

    if not success:
        break


    img = cv2.resize(img, (1088, 1088))
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        #boxes = [ _b  for _b in r.boxes if _b.cls[0].cpu().numpy().astype('int') == 32]
        print(f"dq is {dq}")

        print('boxes:',[_b.cls[0].cpu().numpy().astype('int') for _b in boxes])

        if len(boxes)>1 and z!=0 and len(dq)>0:
            a,b,c=nearest(boxes,dq[0])
            print(a,b,c)
            if b<300:
                boxes=boxes[c]
        for box in boxes:
            #import pdb ; pdb.set_trace()
            if len(dq)>0:
                a,b,c=nearest(box,dq[0])
                print("dis==",b)
                if b>300:
                    continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
            conf=box.conf[0].cpu().numpy()
            if conf > 0.5:
                cv2.circle(img, tuple((int((x1 + x2) / 2), int((y1 + y2) / 2))), 15, (255, 0, 0), 2)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                dq.appendleft(center)
                cv2.putText(img, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for i in range(1, len(dq)):
                if dq[i - 1] is None or dq[i] is None:
                    continue
                cv2.line(img, dq[i - 1], dq[i], (0, 0, 255), thickness=5)

    writer.write(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
