from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import time

# Создаем пустое изображение размером 1920x1080
img = Image.new('RGB', (640, 640), color = 'white')
d = ImageDraw.Draw(img)
ball_coords = []
# Читаем файл лога
with open('ball.log', 'r') as f:
    for line in f:
        # Разбиваем строку на части
        frame, coords, ball_class = line.strip().split(';')
        # Преобразуем координаты в кортеж
        x, y = map(int, coords.strip('()').split(','))
        # Добавляем координаты мяча в список
        ball_coords.append((x, y))
        # Удаляем первую координату, если список содержит более 10 элементов
        if len(ball_coords) > 10:
            ball_coords.pop(0)
        
        # Очищаем изображение
        d.rectangle((0, 0, 1920, 1080), fill='white')
        
        # Рисуем мячи на изображении
        for x, y in ball_coords:
            d.ellipse((x-5, y-5, x+5, y+5), fill='red')
        
        # Преобразуем изображение в массив numpy
        img_array = np.array(img)
        
        # Отображаем изображение
        plt.imshow(img_array)
        plt.show(block=False)
        plt.pause(0.01)  # Пауза для отображения изображения
        plt.clf()  # Очищаем текущую фигуру
        
        time.sleep(0.02)  # Пауза для имитации реального времени
