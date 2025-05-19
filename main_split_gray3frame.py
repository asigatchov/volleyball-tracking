from ultralytics import YOLO
import numpy as np
import cv2
from helper import create_video_writer
from collections import deque
import time
import argparse
import json
from src.ball_tracker import BallTracker, Track
import os




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
file_tracks = '.'.join(video_file.split('.')[:-1]) + '.txt'
if os.path.exists(file_tracks):
    os.remove(file_tracks)

# file_model = "yolov10s.pt"
model = YOLO(file_model) # model name
model.to('cuda')

cap = cv2.VideoCapture(video_file) # file name
# cap = cv2.VideoCapture("video/game_p1_006.mp4") # file name

writer = create_video_writer(cap, "Output6.mp4")  # output file name

fps = int(cap.get(cv2.CAP_PROP_FPS))
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



def calculate_iou(box1, box2):
    """
    Вычисляет IoU (Intersection over Union) между двумя боксами.
    Каждый бокс представлен в виде словаря с ключами x1, y1, x2, y2.
    """
    # Координаты пересечения
    x_left = max(box1["x1"], box2["x1"])
    y_top = max(box1["y1"], box2["y1"])
    x_right = min(box1["x2"], box2["x2"])
    y_bottom = min(box1["y2"], box2["y2"])
    
    # Проверка на пересечение
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Площадь пересечения
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Площади боксов
    box1_area = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    box2_area = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    
    # IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def merge_overlapping_boxes(boxes, iou_threshold=0.3):
    """
    Объединяет перекрывающиеся боксы одного класса в один средний бокс.
    
    Args:
        boxes: Список словарей с ключами x1, y1, x2, y2, confidence
        iou_threshold: Порог IoU для определения перекрытия боксов
        
    Returns:
        Список объединенных боксов
    """
    if not boxes:
        return []
    
    # Сортируем боксы по уверенности (confidence) в порядке убывания
    sorted_boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
    merged_boxes = []
    
    while sorted_boxes:
        # Берем бокс с наибольшей уверенностью
        current_box = sorted_boxes.pop(0)
        boxes_to_merge = [current_box]
        i = 0
        
        # Находим все боксы, которые перекрываются с текущим
        while i < len(sorted_boxes):
            if calculate_iou(current_box, sorted_boxes[i]) > iou_threshold:
                boxes_to_merge.append(sorted_boxes.pop(i))
            else:
                i += 1
        
        # Если нашли перекрывающиеся боксы, объединяем их
        if len(boxes_to_merge) > 1:
            # Вычисляем средние координаты и уверенность
            sum_x1 = sum(box["x1"] for box in boxes_to_merge)
            sum_y1 = sum(box["y1"] for box in boxes_to_merge)
            sum_x2 = sum(box["x2"] for box in boxes_to_merge)
            sum_y2 = sum(box["y2"] for box in boxes_to_merge)
            sum_conf = sum(box["confidence"] for box in boxes_to_merge)
            
            count = len(boxes_to_merge)
            merged_box = {
                "x1": int(sum_x1 / count),
                "y1": int(sum_y1 / count),
                "x2": int(sum_x2 / count),
                "y2": int(sum_y2 / count),
                "confidence": sum_conf / count  # Средняя уверенность
            }
            merged_boxes.append(merged_box)
        else:
            # Если нет перекрывающихся боксов, добавляем текущий бокс как есть
            merged_boxes.append(current_box)
    
    return merged_boxes

def save_detections_json4frame(frame_num, detections):
    json_dir = "jsons/"
    os.makedirs(json_dir, exist_ok=True)
    with open(os.path.join(json_dir, f"frame_{frame_num}.json"), "w") as f:
        json.dump(detections, f)

def process_two_regions(img, model, threshold=0.6 ):
    """
    Обрабатывает два участка 1024x1024: один слева, другой справа, выравнивая их по краям.
    Возвращает результаты детекции в пространстве координат 1920x1080.
    Объединяет перекрывающиеся боксы одного класса в один средний бокс.
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

        #boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype('int') == 0]

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
                    'cls_id':0,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": float(conf)
                })
    
    # Объединяем перекрывающиеся боксы одного класса
    merged_objects = merge_overlapping_boxes(detected_objects, iou_threshold=0.1)
    
    return merged_objects

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

tracker = BallTracker(buffer_size=1500)



while True:
    z += 1
    print(z)
    success, img = cap.read()
    if not success:
        break

    frame_num += 1

    # if frame_num % 2 == 0:
    #     continue

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
    save_detections_json4frame(frame_num, detected_objects)
    
    cv2.putText(img, f'frame: {frame_num:09d}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)

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


    main_id, tracks, deleted_tracks = tracker.update(detected_objects,frame_num)


    if len(deleted_tracks) > 0:
        with open(file_tracks, 'a') as f:
            for track in deleted_tracks:
                f.write(json.dumps(track.to_dict()) + '\n')

    for track_id, track_data in tracks.items():
        color = (255, 255, 0) if int(track_id) == main_id else (0, 0, 255)
        positions = list(track_data['positions'])

        if len(positions) < 3:
            continue
        print('detection', detections, 'main_id:', main_id)
        print('track', track_data)
        # Рисуем трек
        for i in range(1, len(positions)):

            cv2.line(img,
                    (int(positions[i-1][0][0]), int(positions[i-1][0][1])),
                    (int(positions[i][0][0]), int(positions[i][0][1])),
                    color, 2)

        # Рисуем текущую позицию
        x, y = int(positions[-1][0][0]), int(positions[-1][0][1])
        
        f_diff = track_data['last_frame'] - track_data['start_frame']
        t_time = f_diff / fps
        cv2.putText(img, f'id: {track_id}: {t_time:.2f} {f_diff}', (x+25, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.circle(img,
                  (x,y),
                  10, color, -1)
        
        # Сохраняем состояние трекера в JSON-файл каждые 100 кадров
        if frame_num % 100 == 0:
            try:
                with open(f"tracker_state_{frame_num}.json", "w") as f:
                    f.write(tracker.to_json())
                print(f"Saved tracker state to tracker_state_{frame_num}.json")
            except Exception as e:
                print(f"Error saving tracker state: {e}")

    if main_id is not None:
        writer.write(img)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    # cv2.resizeWindow("Image", 1280, 720)        # Устанавливаем размер окна 1280x720
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
# writer.release()
cv2.destroyAllWindows()
