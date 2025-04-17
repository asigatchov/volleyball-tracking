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

file_model = "models/Volley.ball.yolo11n.pt"
file_model = "yolo11s.pt"
file_model = "runs/detect/train8/weights/best.pt"
file_model = "runs/detect/train_work/train8/weights/best.pt"


# file_model = "models/YaphetL.balltrackernet.pt"
model = YOLO(file_model) # model name
model.to('cuda')

cap = cv2.VideoCapture(video_file) # file name
# cap = cv2.VideoCapture("video/game_p1_006.mp4") # file name

writer = create_video_writer(cap, "Output6.mp4", )  # output file name


def calc_distance(current_point, previous_point, current_frame, previous_frame):
    """
    Проверяет, находится ли расстояние между текущей и предыдущей точкой в пределах допустимого,
    учитывая разницу в номерах кадров.
    """
    if current_point is None or previous_point is None:
        return -1
    frame_diff = abs(current_frame - previous_frame)
    distance = int(np.linalg.norm(np.array(current_point) - np.array(previous_point)))
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
    y_start = 0 # max(0, y_center - crop_height // 2)

    x_end = min(frame_width, x_start + crop_width)
    y_end = crop_height # min(frame_height, y_start + crop_height)

    # Корректируем начало, если конец вышел за границы
    x_start = max(0, x_end - crop_width)
    #y_start = max(0, y_end - crop_height)

    return img[y_start:y_end, x_start:x_end]

def calculate_ball_center_from_queue(dq):
    """
    Рассчитывает средние координаты мяча на основе данных из очереди dq.
    """
    if len(dq) == 0:
        return None
    elif len(dq)== 1:
        return dq[0][0], dq[0][1]

    x_coords = [point[0] for point in list(dq)[:10]]
    y_coords = [point[1] for point in list(dq)[:10]]
    avg_x = int(np.mean(x_coords))
    avg_y = int(np.mean(y_coords))
    return avg_x, avg_y

def smooth_crop_movement(previous_center, current_center, threshold=5):
    """
    Плавно перемещает кадр, если смещение мяча превышает заданный порог.
    """
    if previous_center is None:
        return current_center

    distance = np.linalg.norm(np.array(current_center) - np.array(previous_center))
    if distance < threshold:
        return previous_center

    return current_center

def calculate_velocity_and_acceleration(dq):
    """
    Рассчитывает скорость и ускорение мяча на основе данных из очереди dq.
    """
    if len(dq) < 2:
        return None, None

    # Скорость: разница координат между последними двумя кадрами
    dx = dq[0][0] - dq[1][0]
    dy = dq[0][1] - dq[1][1]
    dt = dq[0][2] - dq[1][2]
    velocity = (dx / dt, dy / dt) if dt != 0 else (0, 0)

    # Ускорение: разница скоростей между последними двумя кадрами
    if len(dq) < 3:
        return velocity, None

    prev_dx = dq[1][0] - dq[2][0]
    prev_dy = dq[1][1] - dq[2][1]
    prev_dt = dq[1][2] - dq[2][2]
    prev_velocity = (prev_dx / prev_dt, prev_dy / prev_dt) if prev_dt != 0 else (0, 0)

    acceleration = (
        (velocity[0] - prev_velocity[0]) / dt,
        (velocity[1] - prev_velocity[1]) / dt,
    ) if dt != 0 else (0, 0)

    return velocity, acceleration


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
        if (
            abs(center[0] - spam_center[0]) <= pixel_tolerance
            and abs(center[1] - spam_center[1]) <= pixel_tolerance
        ):
            spam_list[spam_center] += 1
            if spam_list[spam_center] > threshold:
                print(f"Filtered out stationary ball near {spam_center}")
                return True  # Указывает, что детекция должна быть отфильтрована
            return False

    spam_list[center[:2]] = 1
    return False


def predict_ball_position(dq):
    """
    Прогнозирует положение мяча на основе скорости и ускорения.
    """
    if len(dq) < 2:
        return None

    velocity, acceleration = calculate_velocity_and_acceleration(dq)
    if velocity is None:
        return None

    # Прогнозируем новое положение мяча
    predicted_x = dq[0][0] + velocity[0]
    predicted_y = dq[0][1] + velocity[1]
    return int(predicted_x), int(predicted_y)

# Рассчитываем размеры обрезки
frame_height, frame_width, _ = cap.read()[1].shape
crop_width, crop_height = calculate_crop_dimensions(frame_width, frame_height)

writer = create_video_writer(cap, "Output6.mp4",width=crop_width, height=crop_height)  # output file name


dq = deque(maxlen=15)
dq_predictions = deque(maxlen=15)  # Очередь для детекций и предсказаний

z=0
frame_num = 0
skip_spam = {}
spam_list = {}  # Словарь для хранения неподвижных мячей

previous_ball_center = None  # Переменная для хранения положения мяча с предыдущего кадра

no_detection_count = 0  # Счетчик кадров без детекции

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
        img[0:part_height, part_width : 2 * part_width],  # Верхний средний
        img[0:part_height, 2 * part_width : width],  # Верхний правый
        img[part_height:height, 0:part_width],  # Нижний левый
        img[part_height:height, part_width : 2 * part_width],  # Нижний средний
        img[part_height:height, 2 * part_width : width],  # Нижний правый
    ]

    # Обработка частей батчем
    results = model(sub_images, stream=True)

    # Объединение результатов обратно в один кадр
    detected = False  # Флаг для проверки наличия детекции
    for idx, r in enumerate(results):
        boxes = r.boxes
        y_offset = 0 if idx < 3 else part_height
        x_offset = (idx % 3) * part_width

        boxes = [_b for _b in boxes if _b.cls[0].cpu().numpy().astype("int") == 0]

        # import pdb ; pdb.set_trace()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype("int")
            conf = box.conf[0].cpu().numpy()

            if conf > 0.50:
                # Смещение координат обратно в общий кадр
                x1 += x_offset
                x2 += x_offset
                y1 += y_offset
                y2 += y_offset

                radius = int((x2 - x1) / 2) + 1
                center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                    frame_num,
                )  # Add frame number to center
                dicstance = 0
                dist_frame = 1
                if len(dq) > 0:
                    dist_frame = abs(dq[0][2] - center[2])
                    dicstance = calc_distance(
                        dq[0][:2], center[:2], dq[0][2], center[2]
                    )  # Pass frame numbers
                # time.sleep(0.1)
                cv2.putText(
                    img,
                    f"dist: {dicstance:.3f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                cv2.circle(img, tuple(center[:2]), radius, (255, 0, 0), 2)

                cv2.putText(
                    img,
                    f"{conf:.2f} r: {radius}",
                    (center[0] - 10, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                if filter_false_detections(center, spam_list):
                    continue  # Пропускаем неподвижные мячи


                if skip_spam.get(center[:2]) is None:
                    skip_spam[center[:2]] = 1

                skip_spam[center[:2]] += 1
                print("coord:", center, skip_spam[center[:2]])

                detected = True  # Устанавливаем флаг, если есть детекция
                no_detection_count = 0  # Сбрасываем счетчик при детекции
                if int(dicstance) < (50 * dist_frame):
                    dq.appendleft(center)  # Добавляем только детекции в dq
                    dq_predictions.appendleft(center)  # Добавляем детекции в dq_predictions

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

                cv2.putText(
                    img,
                    f"speed_x: {speed_x:.2f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
                cv2.putText(
                    img,
                    f"accel_x: {acceleration_x:.2f}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    1,
                )

                if int(dicstance) < (50 * dist_frame):
                    pass
                else:
                    if len(dq) > 1:
                        dq.pop()
            else:
                pass

        for i in range(1, len(dq)):
            if dq[i - 1] is None or dq[i] is None:
                continue
            cv2.line(img, dq[i - 1][:2], dq[i][:2], (0, 0, 255), thickness=5)
            cv2.putText(
                img,
                f"Ball: {dq[i][:2]}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

    if not detected:
        no_detection_count += 1
        if no_detection_count > 4:
            print("No detection for 4 frames, stopping predictions.")
        else:
            if len(dq) > 0:
                predicted_position = predict_ball_position(dq)
                if predicted_position:
                    dq_predictions.appendleft(
                        (predicted_position[0], predicted_position[1], frame_num)
                    )  # Добавляем предсказания в dq_predictions
                    cv2.circle(img, predicted_position, 10, (0, 255, 255), 2)
                    cv2.putText(
                        img,
                        f"Predicted",
                        (predicted_position[0] - 10, predicted_position[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

    # Визуализация очереди детекций (красным, ширина 5px)
    for i in range(1, len(dq)):
        if dq[i - 1] is None or dq[i] is None:
            continue
        cv2.line(img, dq[i - 1][:2], dq[i][:2], (0, 0, 255), thickness=9)
        cv2.putText(
            img,
            f"Ball: {dq[i][:2]}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    # Визуализация очереди детекций и предсказаний (желтым, ширина 9px)
    for i in range(1, len(dq_predictions)):
        if dq_predictions[i - 1] is None or dq_predictions[i] is None:
            continue
        cv2.line(
            img,
            dq_predictions[i - 1][:2],
            dq_predictions[i][:2],
            (0, 255, 255),
            thickness=5,
        )

    # Рассчитываем координаты мяча только на основе данных из очереди dq
    ball_center = calculate_ball_center_from_queue(dq)

    if ball_center:
        # Плавное движение кадра по мячу
        ball_center = smooth_crop_movement(previous_ball_center, ball_center)
        previous_ball_center = ball_center
    else:
        # Прогнозируем положение мяча, если его нет в текущем кадре
        ball_center = predict_ball_position(dq)
        if ball_center:
            previous_ball_center = ball_center

    if ball_center:
        img = crop_frame_to_ball(img, ball_center, crop_width, crop_height)
        cv2.putText(img, f'Ball: {ball_center}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Вычисляем скорость и ускорение для отображения
    velocity, acceleration = calculate_velocity_and_acceleration(dq)
    if velocity:
        cv2.putText(img, f'Velocity: {velocity}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    if acceleration:
        cv2.putText(img, f'Acceleration: {acceleration}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    writer.write(img)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
    # cv2.resizeWindow("Image", 1280, 720)        # Устанавливаем размер окна 1280x720
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
# writer.release()
cv2.destroyAllWindows()
