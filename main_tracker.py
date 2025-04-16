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
#model = YOLO("yolo-default/yolo11s.pt") # model name


file_model = "runs/detect/train7/weights/best.pt"

#file_model = "runs/detect/train7/weights/best.pt"
model = YOLO(file_model) # model name
model.to('cuda')
мяч = [0,0]

#cap = cv2.VideoCapture("video/p4/game_005.mp4") # file name
#cap = cv2.VideoCapture("video/man_pro.mp4") # file name
#cap = cv2.VideoCapture("video/game_019.mp4") # file name

video_file = 'video/p4/game_005.mp4'
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
    results = model.track(img, persist=True)

        # Visualize the results on the frame
    annotated_frame = results[0].plot()

        # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()