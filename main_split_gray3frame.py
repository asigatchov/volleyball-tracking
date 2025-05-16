from ultralytics import YOLO
import numpy as np
import cv2
from helper import create_video_writer
from collections import deque
import time
import argparse
from src.ball_tracker import BallTracker




# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")

parser.add_argument("video_file", type=str, help="Path to the video file")
parser.add_argument(
        "--model_primary_path",
        "-mp",
        type=str,
        required=True,
        help="Path to the primary YOLO model",
    )
args = parser.parse_args()

video_file = args.video_file  # Получение пути к видеофайлу из аргументов командной строки
file_model = args.model_primary_path  # Путь к модели

# file_model = "yolov10s.pt"
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

def process_image_parts(img, model ,threshold=0.9):
    """
    Разделяет изображение на 6 частей, обрабатывает их моделью и возвращает найденные объекты
    с пересчитанными координатами для полного изображения.
    """
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
    detected_objects = []
    for idx, r in enumerate(results):
        boxes = r.boxes
        y_offset = 0 if idx < 3 else part_height
        x_offset = (idx % 3) * part_width

        boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype('int') == 0]

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')

            if abs((2 - x1) * (y2 - y1)) < 100:
                continue

            conf = box.conf[0].cpu().numpy()
            if conf > threshold:
                # Смещение координат обратно в общий кадр
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                detected_objects.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf
                })

    return detected_objects

def process_two_regions(img, model, threshold=0.6):
    """
    Обрабатывает два участка 1024x1024: один слева, другой справа, выравнивая их по краям.
    Возвращает результаты детекции в пространстве координат 1920x1080.
    """
    height, width, _ = img.shape

    scale =  width / 1920

    region_size = int(1024 * scale)
    vertical_offset = (height - region_size) // 2  # Отступ сверху и снизу

    # Рисуем горизонтальные красные линии с отступом сверху и снизу
    cv2.line(img, (0, vertical_offset), (width, vertical_offset), (0, 0, 255), 2)  # Верхняя линия
    cv2.line(img, (0, height - vertical_offset), (width, height - vertical_offset), (0, 0, 255), 2)  # Нижняя линия

    # Определяем два региона
    left_region = img[vertical_offset:vertical_offset + region_size, 0:region_size]
    right_region = img[vertical_offset:vertical_offset + region_size, width - region_size:width]

    print("Left region shape:", left_region.shape)
    print("Right region shape:", right_region.shape)
    # Обрабатываем регионы моделью
    results = model([left_region, right_region], stream=True)

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    detected_objects = []
    for idx, r in enumerate(results):
        boxes = r.boxes
        x_offset = 0 if idx == 0 else width - region_size  # Смещение: 0 для левого региона, width - region_size для правого
        y_offset = vertical_offset

        boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype('int') == 0]

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype('int')

            conf = box.conf[0].cpu().numpy()
            if conf > threshold:
                # Смещение координат обратно в общий кадр
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                detected_objects.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf
                })

    return detected_objects

frame_queue = deque(maxlen=3)  # Очередь для хранения трех последовательных кадров

def create_detection_frame(queue):
    """
    Создает новый кадр для детекции из трех последовательных кадров.
    Красный канал - первый кадр, зеленый - второй, синий - третий.
    """
    if len(queue) < 3:
        return None
    r = queue[0]
    g = queue[1]
    b = queue[2]
    return cv2.merge((r,g,b))  # Объединяем каналы в один кадр

tracker = BallTracker(buffer_size=15)



while True:
    z += 1
    print(z)
    success, img = cap.read()
    if not success:
        break

    frame_num += 1

    # Преобразуем кадр в grayscale и добавляем в очередь
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_queue.append(gray_frame)

    # Создаем новый кадр для детекции
    detection_frame = create_detection_frame(frame_queue)
    if detection_frame is None:
        continue  # Пропускаем, если недостаточно кадров

    # Используем новый кадр для обработки
    detected_objects = process_two_regions(detection_frame, model, 0.65)

    detected = False
    print("Detected objects:", detected_objects)
    # Обработка найденных объектов

    detections = []
    for obj in detected_objects:
        x1, y1, x2, y2, conf = obj["x1"], obj["y1"], obj["x2"], obj["y2"], obj["confidence"]
        radius = int((x2 - x1) / 2) + 1
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2), frame_num)
        detections.append(center[:2])
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

   
    main_id, tracks = tracker.update(detections,frame_num  )

    print('detection', detections, 'main_id:', main_id)
    for track_id, track in tracks.items():
        color = (255, 255, 0) if track_id == main_id else (0, 0, 255)
        positions = list(track['positions'])
        
        if len(positions) < 3:
            continue
        print('detection', detections, 'main_id:', main_id)
        print('track', track)
        # Рисуем трек
        for i in range(1, len(positions)):

            cv2.line(img, 
                    (int(positions[i-1][0][0]), int(positions[i-1][0][1])),
                    (int(positions[i][0][0]), int(positions[i][0][1])),
                    color, 2)
            
        # Рисуем текущую позицию
        x, y = int(positions[-1][0][0]), int(positions[-1][0][1])
        cv2.putText(img, f'id: {track_id}', (x+25, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.circle(img, 
                  (x,y), 
                  10, color, -1)


    #     detected = True  # Устанавливаем флаг, если есть детекция
    #     no_detection_count = 0  # Сбрасываем счетчик при детекции

    #     if dicstance < (30 * dist_frame):
    #         # import pdb; pdb.set_trace()
    #         dq.appendleft(center)  # Добавляем только детекции в dq
    #         dq_predictions.appendleft(center)  # Добавляем детекции в dq_predictions
    #         with open("ball.log", "a") as file:
    #             file.write(f"{frame_num};{center[0]};{center[1]};{radius}\n")
    #     else:
    #         if len(dq) > 1:
    #             dq.pop()

    #     speed_x = 0
    #     acceleration_x = 0
    #     if len(dq) > 0:
    #         prev_x, _, prev_frame = dq[0]
    #         frame_diff = frame_num - prev_frame
    #         if frame_diff > 0:
    #             speed_x = (center[0] - prev_x) / frame_diff
    #             if len(dq) > 1:
    #                 prev_speed_x = (prev_x - dq[1][0]) / (prev_frame - dq[1][2])
    #                 acceleration_x = (speed_x - prev_speed_x) / frame_diff

    #     cv2.putText(img, f'speed_x: {speed_x:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    #     cv2.putText(img, f'accel_x: {acceleration_x:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # if not detected:
    #     no_detection_count += 1
    #     if no_detection_count > 4:
    #         print("No detection for 4 frames, stopping predictions.")
    #     else:
    #         if len(dq) > 0:
    #             predicted_position = predict_position(dq)
    #             if predicted_position:
    #                 dq_predictions.appendleft((predicted_position[0], predicted_position[1], frame_num))  # Добавляем предсказания в dq_predictions
    #                 cv2.circle(img, predicted_position, 10, (0, 255, 255), 2)
    #                 cv2.putText(img, f'Predicted', (predicted_position[0] - 10, predicted_position[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # # Визуализация очереди детекций (красным, ширина 5px)
    # for i in range(1, len(dq)):
    #     if dq[i - 1] is None or dq[i] is None:
    #         continue
    #     cv2.line(img, dq[i - 1][:2], dq[i][:2], (0, 0, 255), thickness=9)
    #     cv2.putText(img, f'Ball: {dq[i][:2]}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # # Визуализация очереди детекций и предсказаний (желтым, ширина 9px)
    # for i in range(1, len(dq_predictions)):
    #     if dq_predictions[i - 1] is None or dq_predictions[i] is None:
    #         continue
    #     cv2.line(img, dq_predictions[i - 1][:2], dq_predictions[i][:2], (0, 255, 255), thickness=5)

    writer.write(img)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    # cv2.resizeWindow("Image", 1280, 720)        # Устанавливаем размер окна 1280x720
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
# writer.release()
cv2.destroyAllWindows()
