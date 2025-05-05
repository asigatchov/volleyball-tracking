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
# model = YOLO("yolo-default/yolo11s.pt") # model name

file_model = "models/asigatchov/yolo11x_ball_10kimg_640x540_e300_20250428.pt"
# file_model = (
#  "models/asigatchov/yolo11s_ball_10kimg_640x540_e300_20250428.pt"
# )

file_model = "yolo11s-obb.pt"

file_model = "runs/detect/train21/weights/best.pt"

model = YOLO(file_model) # model name
model.to('cuda')

cap = cv2.VideoCapture(video_file) # file name
# cap = cv2.VideoCapture("video/game_p1_006.mp4") # file name

writer = create_video_writer(cap, "Output6.mp4")  # output file name


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

def predict_position(dq):
    """
    Прогнозирует положение мяча на основе предыдущих данных.
    """
    if len(dq) < 2:
        return None
    x1, y1, frame1 = dq[0]
    x2, y2, frame2 = dq[1]
    frame_diff = frame1 - frame2
    if frame_diff == 0:
        return None
    speed_x = (x1 - x2) / frame_diff
    speed_y = (y1 - y2) / frame_diff
    predicted_x = x1 + speed_x
    predicted_y = y1 + speed_y
    return (int(predicted_x), int(predicted_y))  # Прогнозируем x и y

def filter_false_detections(center, spam_list, threshold=3, pixel_tolerance=7):
    """
    Фильтрует ложные детекции и неподвижные мячи.
    Если мяч неподвижен и постоянно детектируется, добавляет его в спам-лист.
    Также фильтрует детекции по квадрату в 5 пикселей.
    """
    for spam_center in spam_list:
        if abs(center[0] - spam_center[0]) <= pixel_tolerance and abs(center[1] - spam_center[1]) <= pixel_tolerance:
            spam_list[spam_center] += 1
            if spam_list[spam_center] > threshold:
                print(f"Filtered out stationary ball near {spam_center}")
                return True  # Указывает, что детекция должна быть отфильтрована
            return False

    spam_list[center[:2]] = 1
    return False

dq = deque(maxlen=15)  # Очередь для детекций мяча
dq_predictions = deque(maxlen=15)  # Очередь для детекций и предсказаний
z = 0
frame_num = 0
skip_spam = {}
no_detection_count = 0  # Счетчик кадров без детекции
spam_list = {}  # Словарь для хранения неподвижных мячей

def preprocess_frame(frame):
    # Улучшение контраста
    #frame = cv2.equalizeHist(frame)
    # Размытие для уменьшения шума
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    return frame

while True:
    z += 1
    print(z)
    success, img = cap.read()
    if not success:
        break

    frame_num += 1

    img = preprocess_frame(img)
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

    # Рисуем черные линии для визуализации разделения
    for i in range(1, 3):  # Вертикальные линии
        cv2.line(img, (i * part_width, 0), (i * part_width, height), (0, 0, 0), 2)
    for i in range(1, 2):  # Горизонтальная линия
        cv2.line(img, (0, i * part_height), (width, i * part_height), (0, 0, 0), 2)

    # Обработка частей батчем
    results = model(sub_images, stream=True)

    # Объединение результатов обратно в один кадр
    detected = False  # Флаг для проверки наличия детекции
    for idx, r in enumerate(results):
        boxes = r.boxes
        y_offset = 0 if idx < 3 else part_height
        x_offset = (idx % 3) * part_width

        boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype('int') == 0]

        # import pdb ; pdb.set_trace()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')

            if abs((2 - x1) * (y2 - y1)) < 100:
                continue

            conf = box.conf[0].cpu().numpy()
            if conf >0.60:
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
                # time.sleep(0.1)
                cv2.putText(img, f'dist: {dicstance:.3f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.circle(img, tuple(center[:2]), radius, (255, 0, 0), 2)

                cv2.putText(img, f'{conf:.2f} r: {radius}', (center[0] - 10, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if filter_false_detections(center, spam_list):
                    continue  # Пропускаем неподвижные мячи

                detected = True  # Устанавливаем флаг, если есть детекция
                no_detection_count = 0  # Сбрасываем счетчик при детекции

                if dicstance < (30 * dist_frame):
                    # import pdb; pdb.set_trace()
                    dq.appendleft(center)  # Добавляем только детекции в dq
                    dq_predictions.appendleft(center)  # Добавляем детекции в dq_predictions
                    with open("ball.log", "a") as file:
                        file.write(f"{frame_num};{center[:2]};\n")
                else:
                    if len(dq) > 1:
                        dq.pop()

                speed_x = 0
                acceleration_x = 0
                if len(dq) > 0:
                    prev_x, _, prev_frame = dq[0]
                    frame_diff = frame_num - prev_frame
                    if frame_diff > 0:
                        speed_x = (center[0] - prev_x) / frame_diff
                        if len(dq) > 1:
                            prev_speed_x = (prev_x - dq[1][0]) / (prev_frame - dq[1][2])
                            acceleration_x = (speed_x - prev_speed_x) / frame_diff

                cv2.putText(img, f'speed_x: {speed_x:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(img, f'accel_x: {acceleration_x:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            else:
                pass

        for i in range(1, len(dq)):
            if dq[i - 1] is None or dq[i] is None:
                continue
            cv2.line(img, dq[i - 1][:2], dq[i][:2], (0, 0, 255), thickness=5)
            cv2.putText(img, f'Ball: {dq[i][:2]}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    if not detected:
        no_detection_count += 1
        if no_detection_count > 4:
            print("No detection for 4 frames, stopping predictions.")
        else:
            if len(dq) > 0:
                predicted_position = predict_position(dq)
                if predicted_position:
                    dq_predictions.appendleft((predicted_position[0], predicted_position[1], frame_num))  # Добавляем предсказания в dq_predictions
                    cv2.circle(img, predicted_position, 10, (0, 255, 255), 2)
                    cv2.putText(img, f'Predicted', (predicted_position[0] - 10, predicted_position[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Визуализация очереди детекций (красным, ширина 5px)
    for i in range(1, len(dq)):
        if dq[i - 1] is None or dq[i] is None:
            continue
        cv2.line(img, dq[i - 1][:2], dq[i][:2], (0, 0, 255), thickness=9)
        cv2.putText(img, f'Ball: {dq[i][:2]}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Визуализация очереди детекций и предсказаний (желтым, ширина 9px)
    for i in range(1, len(dq_predictions)):
        if dq_predictions[i - 1] is None or dq_predictions[i] is None:
            continue
        cv2.line(img, dq_predictions[i - 1][:2], dq_predictions[i][:2], (0, 255, 255), thickness=5)

    writer.write(img)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    # cv2.resizeWindow("Image", 1280, 720)        # Устанавливаем размер окна 1280x720
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
# writer.release()
cv2.destroyAllWindows()
