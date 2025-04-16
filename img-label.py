import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Переменные для хранения состояния
current_image = None
start_point = None
end_point = None
bbox_coordinates = None

def load_image(input_image):
    global current_image, start_point, end_point, bbox_coordinates
    # Сброс состояния при загрузке нового изображения
    start_point = None
    end_point = None
    bbox_coordinates = None
    # Конвертируем в PIL Image для единообразия
    if isinstance(input_image, np.ndarray):
        current_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    else:
        current_image = input_image
    return current_image

def select_region(event: gr.SelectData):
    global start_point, end_point, bbox_coordinates
    start_point = (event.index[0], event.index[1])
    end_point = (event.index[0], event.index[1])
    bbox_coordinates = None
    return update_image()

def move_region(event: gr.SelectData):
    global end_point
    if start_point is not None:
        end_point = (event.index[0], event.index[1])
        return update_image()
    return None

def release_region(event: gr.SelectData):
    global end_point, bbox_coordinates
    if start_point is not None:
        end_point = (event.index[0], event.index[1])
        # Убедимся, что координаты упорядочены (x1 < x2, y1 < y2)
        x1, y1 = start_point
        x2, y2 = end_point
        bbox_coordinates = {
            "x1": min(x1, x2),
            "y1": min(y1, y2),
            "x2": max(x1, x2),
            "y2": max(y1, y2),
            "width": abs(x2 - x1),
            "height": abs(y2 - y1)
        }
        return update_image(), str(bbox_coordinates)
    return None, ""

def update_image():
    global current_image, start_point, end_point
    if current_image is None:
        return None
    
    img = current_image.copy()
    draw = ImageDraw.Draw(img)
    
    if start_point and end_point:
        # Рисуем прямоугольник от start_point до end_point
        draw.rectangle([start_point, end_point], outline="red", width=2)
    
    return img

with gr.Blocks() as demo:
    gr.Markdown("## Выделение объекта на изображении")
    gr.Markdown("1. Загрузите изображение 2. Кликните и тяните для выделения области")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Загрузите изображение")
            load_button = gr.Button("Загрузить изображение")
        
        with gr.Column():
            output_image = gr.Image(label="Выделенная область", interactive=True)
            coordinates_output = gr.Textbox(label="Координаты выделенной области")
    
    # Обработчики событий
    load_button.click(load_image, inputs=input_image, outputs=output_image)
    
    # Обработчики для выделения области
    output_image.select(select_region, None, output_image)
    output_image.select(move_region, None, output_image, show_progress=False)
    output_image.select(release_region, None, [output_image, coordinates_output], show_progress=False)

if __name__ == "__main__":
    demo.launch()
