import cv2
import os
import numpy as np

# Глобальные переменные
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
img = None
img_copy = None
current_idx = 0
image_files = []
folder_path = ""

def load_images_from_folder(folder):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    return [f for f in os.listdir(folder) if f.lower().endswith(valid_extensions)]

def load_image(idx):
    global img, img_copy, ix, iy, fx, fy, drawing
    if 0 <= idx < len(image_files):
        img_path = os.path.join(folder_path, image_files[idx])
        img = cv2.imread(img_path)
        
        if img is not None:
            img_copy = img.copy()
            ix, iy = -1, -1
            fx, fy = -1, -1
            drawing = False
            print(f"\nЗагружено изображение {idx+1}/{len(image_files)}: {image_files[idx]}")
            return True
    return False

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        img_copy = img.copy()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
            temp_img = img_copy.copy()
            cv2.rectangle(temp_img, (ix, iy), (fx, fy), (0, 255, 0), 2)
            cv2.imshow('Image', temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        cv2.rectangle(img, (ix, iy), (fx, fy), (0, 255, 0), 2)
        cv2.imshow('Image', img)
        
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)
        width = x2 - x1
        height = y2 - y1
        
        print("\nКоординаты выделенной области:")
        print(f"Левый верхний угол: ({x1}, {y1})")
        print(f"Правый нижний угол: ({x2}, {y2})")
        print(f"Ширина: {width}, Высота: {height}")

def show_help():
    print("\nУправление:")
    print("n - следующее изображение")
    print("p - предыдущее изображение")
    print("r - сбросить выделение")
    print("s - сохранить координаты в файл")
    print("q или ESC - выход")

def save_coordinates():
    if ix == -1 or iy == -1 or fx == -1 or fy == -1:
        print("Нет выделенной области для сохранения")
        return
    
    x1, y1 = min(ix, fx), min(iy, fy)
    x2, y2 = max(ix, fx), max(iy, fy)
    
    filename = image_files[current_idx]
    output_file = os.path.splitext(filename)[0] + "_coords.txt"
    
    with open(output_file, 'w') as f:
        f.write(f"Изображение: {filename}\n")
        f.write(f"Левый верхний угол: ({x1}, {y1})\n")
        f.write(f"Правый нижний угол: ({x2}, {y2})\n")
        f.write(f"Ширина: {x2-x1}, Высота: {y2-y1}\n")
    
    print(f"Координаты сохранены в файл: {output_file}")

def main():
    global current_idx, folder_path, image_files, img
    
    folder_path = input("Введите путь к каталогу с изображениями: ")
    if not os.path.isdir(folder_path):
        print("Ошибка: указанный путь не является каталогом")
        return
    
    image_files = load_images_from_folder(folder_path)
    if not image_files:
        print("В каталоге не найдено изображений")
        return
    
    image_files.sort()
    print(f"Найдено {len(image_files)} изображений")
    current_idx = 0
    if not load_image(current_idx):
        print("Ошибка при загрузке первого изображения")
        return
    
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle)
    show_help()

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n'):  # Следующее изображение
            if current_idx < len(image_files) - 1:
                current_idx += 1
                if not load_image(current_idx):
                    current_idx -= 1
        
        elif key == ord('p'):  # Предыдущее изображение
            if current_idx > 0:
                current_idx -= 1
                if not load_image(current_idx):
                    current_idx += 1
        
        elif key == ord('r'):  # Сбросить выделение
            img = img_copy.copy()
            ix, iy = -1, -1
            fx, fy = -1, -1
            print("Выделение сброшено")
        
        elif key == ord('s'):  # Сохранить координаты
            save_coordinates()
        
        elif key == ord('h'):  # Показать справку
            show_help()
        
        elif key == ord('q') or key == 27:  # Выход
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
