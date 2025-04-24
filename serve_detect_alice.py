import pandas as pd
import numpy as np


def load_data(filename):
    """
    Загружает данные из CSV файла
    """
    # Проверяем, есть ли заголовок в файле
    with open(filename, "r") as file:
        first_line = file.readline().strip()
        if not first_line.startswith("frame_num;x;y"):
            # Добавляем заголовок, если его нет
            data = pd.read_csv(filename, delimiter=";", header=None, names=["frame_num", "x", "y", "cnt"])
        else:
            data = pd.read_csv(filename, delimiter=";")
    return data


def calculate_velocity(data):
    """
    Вычисляет скорость мяча между кадрами
    """
    # Вычисляем разницу между последовательными кадрами
    dx = np.diff(data["x"])
    dy = np.diff(data["y"])

    # Вычисляем расстояние между кадрами
    distance = np.sqrt(dx**2 + dy**2)

    # Умножаем на частоту кадров (30 fps)
    velocity = distance * 30

    return velocity


def detect_serve(data, velocity_threshold=1.5, y_threshold=0.7):
    """
    Определяет момент подачи мяча

    velocity_threshold - порог скорости для определения движения
    y_threshold - порог по Y для определения верхней части кадра (0-1)
    """
    # Находим максимальную Y координату
    max_y = data["y"].max()

    # Определяем верхнюю границу для поиска подачи
    upper_bound = max_y * y_threshold

    # Находим кадры, где мяч выше границы
    candidates = data[data["y"] > upper_bound]

    # Вычисляем скорость для кандидатов
    candidate_velocities = calculate_velocity(candidates)

    # Ищем первый кадр с достаточной скоростью
    for i, vel in enumerate(candidate_velocities):
        if abs(vel) > velocity_threshold:
            # Находим индекс в исходных данных
            frame_index = candidates.index[i + 1]
            return frame_index

    return None


def analyze_serve(filename):
    """
    Основная функция анализа подачи
    """
    # Загружаем данные
    data = load_data(filename)

    # Определяем момент подачи
    serve_frame = detect_serve(data)

    if serve_frame is not None:
        print(f"Момент подачи определен: кадр {serve_frame}")
        print(
            f"Координаты: x={data.loc[serve_frame, 'x']}, y={data.loc[serve_frame, 'y']}"
        )
    else:
        print("Момент подачи не определен")


# Пример использования
analyze_serve("tmp_ball.csv")
