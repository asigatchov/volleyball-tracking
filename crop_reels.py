from ultralytics import YOLO
import numpy as np
import cv2
from helper import create_video_writer
from collections import deque
import time
import argparse

# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")
parser.add_argument("video_file", type=str, help="Path to the video file")
args = parser.parse_args()

video_file = args.video_file  # Получение пути к видеофайлу из аргументов командной строки

def nearest(res, init):
    box = res.xyxy.cpu().numpy().astype('int')
    mean_c13 = np.mean(box[:, [0, 2]], axis=1)
    mean_c24 = np.mean(box[:, [1, 3]], axis=1)
    center = np.stack((mean_c13, mean_c24), axis=1).astype('int')
    distances = np.linalg.norm(center - init, axis=1).astype('float')
    return center[np.argmin(distances)], distances[np.argmin(distances)], np.argmin(distances)
#model = YOLO("yolo-default/yolo11s.pt") # model name

file_model = "models/Volley.ball.yolo11n.pt"
file_model = "yolo11s.pt"
file_model = "runs/detect/train8/weights/best.pt"



#file_model = "models/YaphetL.balltrackernet.pt"
model = YOLO(file_model) # model name
model.to('cuda')

cap = cv2.VideoCapture(video_file) # file name
#cap = cv2.VideoCapture("video/game_p1_006.mp4") # file name

writer = create_video_writer(cap, "Output6.mp4", )  # output file name


def calc_distance(current_point, previous_point, current_frame, previous_frame):
    """
    Проверяет, находится ли расстояние между текущей и предыдущей точкой в пределах допустимого,
    учитывая разницу в номерах кадров.
    """
    if current_point is None or previous_point is None:
        return -1
    frame_diff = abs(current_frame - previous_frame)
    distance = np.linalg.norm(np.array(current_point) - np.array(previous_point))
    print('distance:',current_point, previous_point, distance, frame_diff)
    return distance

def calculate_crop_dimensions(frame_width, frame_height):
    """
    Рассчитывает размеры обрезки для соотношения сторон 9:16.
    """
    aspect_ratio = 9 / 16
    if frame_width / frame_height > aspect_ratio:
        new_width = int(frame_height * aspect_ratio)
        new_height = frame_height
    else:
        new_width = frame_width
        new_height = int(frame_width / aspect_ratio)
    return new_width, new_height

def crop_frame_to_ball(img, ball_center, crop_width, crop_height):
    """
    Центрирует кадр по мячу и обрезает его до заданных размеров.
    """
    frame_height, frame_width, _ = img.shape
    x_center, y_center = ball_center

    x_start = max(0, x_center - crop_width // 2)
    y_start = max(0, y_center - crop_height // 2)

    x_end = min(frame_width, x_start + crop_width)
    y_end = min(frame_height, y_start + crop_height)

    # Корректируем начало, если конец вышел за границы
    x_start = max(0, x_end - crop_width)
    y_start = max(0, y_end - crop_height)

    return img[y_start:y_end, x_start:x_end]

def calculate_ball_center_from_queue(dq):
    """
    Рассчитывает средние координаты мяча на основе данных из очереди dq.
    """
    if len(dq) == 0:
        return None
    elif len(dq)== 1:
        return dq[0][0], dq[0][1]

    x_coords = [point[0] for point in list(dq)[:2]]
    y_coords = [point[1] for point in list(dq)[:2]]
    avg_x = int(np.mean(x_coords))
    avg_y = int(np.mean(y_coords))
    return avg_x, avg_y

# Рассчитываем размеры обрезки
frame_height, frame_width, _ = cap.read()[1].shape
crop_width, crop_height = calculate_crop_dimensions(frame_width, frame_height)

writer = create_video_writer(cap, "Output6.mp4",width=crop_width, height=crop_height)  # output file name


dq = deque(maxlen=15)
z=0
frame_num = 0
skip_spam = {}

previous_ball_center = None  # Переменная для хранения положения мяча с предыдущего кадра

while True:
    z += 1
    print(z)
    success, img = cap.read()
    if not success:
        break

    frame_num += 1

 

    # Разделение кадра на 6 частей
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
    results = model(sub_images, stream=True)

    # Объединение результатов обратно в один кадр
    ball_center = None
    for idx, r in enumerate(results):
        boxes = r.boxes
        y_offset = 0 if idx < 3 else part_height
        x_offset = (idx % 3) * part_width


        boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype('int') == 0]

        #import pdb ; pdb.set_trace()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')
            conf = box.conf[0].cpu().numpy()
           
            if conf >0.69:
                # Смещение координат обратно в общий кадр
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                radius = int((x2 - x1) / 2) + 1 
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2), frame_num)  # Add frame number to center
                dicstance = 0
                dist_frame = 1
                if len(dq) > 0:
                    dist_frame = abs(dq[0][2] - center[2])
                    dicstance = calc_distance(dq[0][:2], center[:2], dq[0][2], center[2])  # Pass frame numbers
                #time.sleep(0.1)
                cv2.putText(img, f'dist: {dicstance:.3f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.circle(img, tuple(center[:2]), radius, (255, 0, 0), 2)
               
                cv2.putText(img, f'{conf:.2f} r: {radius}', (center[0] - 10, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if skip_spam.get(center[:2]) is None:
                    skip_spam[center[:2]] = 1

                skip_spam[center[:2]] += 1
                print('coord:', center, skip_spam[center[:2]])  

          
                if skip_spam[center[:2]] < 5 and dicstance  < (30 * dist_frame):
                    #import pdb; pdb.set_trace()
                    dq.appendleft(center)
                    
                    with open('ball.log', 'a') as file:
                        file.write(f'{frame_num};{center[:2]};{skip_spam[center[:2]]}\n')
                else:
                    if len(dq) > 1:
                        dq.pop()
                ball_center = center[:2]  # Сохраняем координаты центра мяча
                break  # Берем только первый найденный мяч
            else:
                pass
               
        for i in range(1, len(dq)):
            if dq[i - 1] is None or dq[i] is None:
                continue
            cv2.line(img, dq[i - 1][:2], dq[i][:2], (0, 0, 255), thickness=5)

    # Рассчитываем координаты мяча только на основе данных из очереди dq
    ball_center = calculate_ball_center_from_queue(dq)

    if ball_center:
        img = crop_frame_to_ball(img, ball_center, crop_width, crop_height)

        writer.write(img)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
   # cv2.resizeWindow("Image", 1280, 720)        # Устанавливаем размер окна 1280x720
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
#writer.release()
cv2.destroyAllWindows()
