import cv2
import json
import numpy as np
from collections import defaultdict
import math
import argparse
import pandas as pd


def load_tracks(log_file):
    tracks = []
    with open(log_file, 'r') as f:
        for line in f:
            track = json.loads(line.strip())
            tracks.append({
                'start_frame': track['start_frame'],
                'last_frame': track['last_frame'],
                'positions': [(pos[0], pos[1], frame) for pos, frame in track['positions']],
                'ball_sizes': track['ball_sizes']
            })
    return tracks


def load_to_dataframe(data, touch_threshold=0.5):
    positions_data = []
    for pos in data['positions']:
        positions_data.append({
            'x': pos[0][0],
            'y': pos[0][1],
            'frame': pos[1]
        })
    
    df = pd.DataFrame(positions_data).sort_values('frame')
    
    # Вычисляем производные движения
    df['v_x'] = df['x'].diff() / df['frame'].diff()
    df['v_y'] = df['y'].diff() / df['frame'].diff()
    df['v_xy'] = np.sqrt(df['v_x']**2 + df['v_y']**2)
    
    df['a_x'] = df['v_x'].diff() / df['frame'].diff()
    df['a_y'] = df['v_y'].diff() / df['frame'].diff()
    df['a_xy'] = np.sqrt(df['a_x']**2 + df['a_y']**2)
    
    df['angle'] = np.degrees(np.arctan2(df['v_y'], df['v_x'])) % 360
    df['angular_velocity'] = df['angle'].diff() / df['frame'].diff()
    
    # Детекция касаний
    df['acceleration_change'] = df['a_xy'].abs()
    df['angle_change'] = abs(df['angular_velocity'])
    
    # Комбинированный показатель изменения траектории
    df['trajectory_change'] = (
        df['acceleration_change'] / df['acceleration_change'].mean() + 
        df['angle_change'] / df['angle_change'].mean()
    )
    
    # Определяем кадры с касаниями
    df['is_touch'] = False
    if len(df) > 1:
        mean_change = df['trajectory_change'].mean()
        std_change = df['trajectory_change'].std()
        threshold = mean_change + touch_threshold * std_change
        
        # Находим кадры с аномальными изменениями траектории
        touch_frames = df[df['trajectory_change'] > threshold]['frame']
        df.loc[df['frame'].isin(touch_frames), 'is_touch'] = True
    
    if 'prediction' in data:
        prediction = data['prediction']
        prediction_frame = df['frame'].max() + 1
        last_row = df.iloc[-1]
        
        pred_df = pd.DataFrame([{
            'x': prediction[0],
            'y': prediction[1],
            'frame': prediction_frame,
            'is_prediction': True,
            'is_touch': False,
            **{k: last_row[k] for k in ['v_x', 'v_y', 'v_xy', 
                                       'a_x', 'a_y', 'a_xy',
                                       'angle', 'angular_velocity']}
        }])
        
        df = pd.concat([df, pred_df], ignore_index=True)
    
    return df




def show_frame_numbers(video_path, frame_numbers):
    """
    Отображает кадры с заданными номерами из видеофайла.
    :param frame_numbers: Список номеров кадров для отображения
    """
    # Проверяем, что список не пуст
    if not frame_numbers:
        print("Список номеров кадров пуст!")
        return
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Ошибка: Не удалось открыть видеофайл!")
        exit()

    current_index = 0  # Текущий индекс в списке frame_numbers

    while True:
        # Получаем номер кадра, который нужно показать
        target_frame = frame_numbers[current_index]

        # Устанавливаем текущую позицию видео (кадр за кадром)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        # Читаем кадр
        ret, frame = cap.read()

        if not ret:
            print(f"Ошибка: Не удалось прочитать кадр {target_frame}!")
            break

        # Выводим номер кадра на изображение
        cv2.putText(
            frame,
            f"Frame: {target_frame}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Показываем кадр
        cv2.imshow("Video Frame", frame)

        # Ждем нажатия любой клавиши
        key = cv2.waitKey(0)

        # Переключаемся на следующий кадр (или выходим)
        current_index += 1
        if current_index >= len(frame_numbers):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_track_frames(video_file, tracks, preview_seconds=1):
    """
    Отображает кадры из видео, соответствующие трекам.

    Args:
        video_file (str): Путь к видеофайлу
        tracks (list): Список словарей с ключами 'start_frame' и 'last_frame'
        preview_seconds (int): Количество секунд до и после трека для отображения
    """
    # Открываем видеофайл
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл {video_file}")
        return

    # Получаем FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Ошибка: Не удалось определить FPS видео")
        cap.release()
        return

    # Конвертируем секунды в кадры
    preview_frames = int(preview_seconds * fps)

    # Сортируем треки по начальному кадру
    sorted_tracks = sorted(tracks, key=lambda x: x["start_frame"])

    # Проходим по всем трекам
    for i, track in enumerate(sorted_tracks):
        start_frame = max(0, track["start_frame"] - preview_frames)
        end_frame = track["last_frame"] + preview_frames

        print(f"\nТрек {i+1}: кадры {track['start_frame']}-{track['last_frame']}")
        print(f"Показываю кадры {start_frame}-{end_frame}")

        # Переходим к начальному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while current_frame <= end_frame and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            # if current_frame % 4 != 0:
            #     continue
            # Добавляем информацию о кадре и треке
            cv2.putText(
                frame,
                f"Frame: {current_frame}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Track: {i+1} ({track['start_frame']}-{track['last_frame']})",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Подсвечиваем кадры внутри трека
            if track["start_frame"] <= current_frame <= track["last_frame"]:
                cv2.rectangle(
                    frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10
                )

            # Показываем кадр
            cv2.imshow(f"Video: {video_file}", frame)

            # Обработка нажатий клавиш
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):  # Выход
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(" "):  # Пауза
                while True:
                    key = cv2.waitKey(0)
                    if key == ord(" "):  # Продолжить воспроизведение
                        break
                    elif key == ord("q"):  # Выход
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord("n"):  # Следующий трек
                        break
                if key == ord("n"):
                    break
            elif key == ord("n"):  # Следующий трек
                break

            current_frame += 1

            # Проверяем, не закончилось ли видео
            if current_frame > end_frame:
                key = cv2.waitKey(2000)  # Задержка 2 секунды между треками
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


def merge_close_tracks(tracks, max_gap=10):
    """
    Объединяет треки, если между last_frame одного трека и start_frame следующего трека меньше max_gap кадров.
    tracks: список треков (dict с ключами 'start_frame', 'last_frame', 'positions', 'ball_sizes')
    Возвращает новый список треков.
    """
    if not tracks:
        return []
    # Сортируем по старту
    tracks = sorted(tracks, key=lambda t: t['start_frame'])
    merged = []
    current = tracks[0].copy()
    for next_track in tracks[1:]:
        if next_track['start_frame'] - current['last_frame'] < max_gap:
            # Объединяем
            current['last_frame'] = next_track['last_frame']
            current['positions'].extend(next_track['positions'])
            if 'ball_sizes' in current and 'ball_sizes' in next_track:
                current['ball_sizes'].extend(next_track['ball_sizes'])
        else:
            merged.append(current)
            current = next_track.copy()
    merged.append(current)
    return merged


def merge_tracks(tracks, max_gap_seconds=4, fps=60):
    max_gap_frames = max_gap_seconds * fps
    merged_tracks = []
    tracks = sorted(tracks, key=lambda x: x['start_frame'])
    
    current_track = tracks[0]
    for next_track in tracks[1:]:
        gap = next_track['start_frame'] - current_track['last_frame']
        if gap <= max_gap_frames:
            # Merge tracks
            current_track['last_frame'] = next_track['last_frame']
            current_track['positions'].extend(next_track['positions'])
            current_track['ball_sizes'].extend(next_track['ball_sizes'])
        else:
            # End of rally, add to merged tracks
            merged_tracks.append(current_track)
            current_track = next_track
    
    merged_tracks.append(current_track)
    return merged_tracks

def filter_short_tracks(tracks, min_length=10):
    """
    Удаляет треки, длина которых (по количеству позиций) меньше min_length.
    """
    return [track for track in tracks if len(track['positions']) >= min_length]

def main():
    # Load and process tracks
    parser = argparse.ArgumentParser(description="Process a video file.")

    parser.add_argument("file_log", type=str, help="Path to the video file")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("-t", '--track_id', type=int, default=-1, help="показать трек")

    args = parser.parse_args()

    file_log = args.file_log
    video_file = args.video_file
    track_id = args.track_id

    print('track_id', track_id)


    #  print(f"Video file: {video_file}")
    #  print(f"Frame numbers: {frame_numbers}")
    #  show_frame_numbers(video_file, frame_numbers)

    tracks = load_tracks(file_log)
    # Объединение близких треков
    #tracks = merge_close_tracks(tracks, max_gap=240)
    m_tracks  = merge_tracks(tracks, max_gap_seconds=4, fps=60) 
    # Фильтрация коротких треков
    #tracks = filter_short_tracks(tracks, min_length=10)
    
    show_track_frames(video_file, m_tracks, preview_seconds=0)


if __name__ == "__main__":
    main()
