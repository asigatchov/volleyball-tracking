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


models = []

model_files = [
    "runs/detect/train24/weights/best.pt",
    "runs/detect/train22/weights/best.pt"
    ]

for file_model in model_files:
    models.append( YOLO(file_model).to('cuda') )  # model name

# Define colors for each model
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

z = 0  # Initialize z

def video_frame_generator(video_file):
    """
    Generator function to read frames from a video file.
    """
    cap = cv2.VideoCapture(video_file)
    while True:
        success, frame = cap.read()
        if not success:
            break
        yield frame
    cap.release()

frame_num = 0  # Initialize frame_num




for img in video_frame_generator(video_file):
    z += 1
    print(z)
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

    # Add model names with colors in the top-left corner
    for i, model_file in enumerate(model_files):
        color = colors[i % len(colors)]  # Assign color to the model
        model_name = model_file.split("/")[-3] # Extract model name from path
        cv2.putText(
            img,
            model_name,
            (10, 30 + i * 20),  # Position each name below the previous
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Process each sub-image with each model
    for i, model in enumerate(models):
        color = colors[i % len(colors)]  # Assign color to the model
        results = model.predict(img, verbose=False)  # Run detection on the full image


        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0]  # Confidence score

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Display confidence score above the box
                label = f"{confidence:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1 + 20 * i, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

    # Display the image with detections

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)  # Allow resizing of the window
    cv2.imshow("Detections", img)
    time.sleep(0.1)  # Add a small delay to control the frame rate
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
