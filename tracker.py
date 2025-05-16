import cv2
from ultralytics import YOLO
import numpy as np
import argparse

from collections import deque

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
    return cv2.merge((r,g, b))  # Объединяем каналы в один кадр



def run_multi_object_tracker(video_path, model_name="yolo11n.pt", tracker_config="botsort.yaml"):
    """
    Выполняет трекинг нескольких объектов на видео с использованием YOLO11.

    Args:
        video_path (str): Путь к видеофайлу или '0' для веб-камеры
        model_name (str): Модель YOLO11 (например, 'yolo11n.pt')
        tracker_config (str): Конфигурация трекера (например, 'botsort.yaml' или 'bytetrack.yaml')
    """
    # Загрузка модели YOLO11
    model = YOLO(model_name)
    frame_queue = deque(maxlen=3)  # Очередь для хранения трех последовательных кадров


    # Открытие видеопотока
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return


    # Получение параметров видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Настройка записи выходного видео
    output_path = "output_tracked_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()

        frame = cv2.resize(frame, (1024,1024))    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        frame_queue.append(gray_frame)

        # Создаем новый кадр для детекции
        detection_frame = create_detection_frame(frame_queue)    
        if detection_frame is None:
            continue

        # Выполнение трекинга
        results = model.track(
            source=detection_frame,
            persist=True,  # Сохранение треков между кадрами
            tracker=tracker_config,
            conf=0.2,  # Порог уверенности
            iou=0.5,   # Порог IoU
        )

        # Обработка результатов
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_names = model.names

            for box, track_id, conf, cls in zip(boxes, ids, confidences, classes):
                x1, y1, x2, y2 = box
                label = f"{class_names[cls]} ID:{track_id} {conf:.2f}"

                # Отрисовка bounding box и метки
                cv2.rectangle(gray_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    gray_frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        # Запись кадра в выходное видео
        out.write(annotated_frame)

        # Отображение кадра (опционально, закомментируйте для headless-систем)
        cv2.imshow("YOLO11 Multi-Object Tracking", gray_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Выходное видео сохранено как {output_path}")

if __name__ == "__main__":
    # Пример использования
    run_multi_object_tracker(video_file, model_name=file_model, tracker_config="botsort.yaml")