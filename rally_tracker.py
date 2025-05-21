import json
import cv2
import numpy as np
from collections import defaultdict
import math
import argparse
import os

def ensure_reels_dir():
    if not os.path.exists('reels'):
        os.makedirs('reels')

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

def calculate_velocities(pos1, pos2, frame_diff, fps=60):
    """Calculate velocities for each axis.
    Returns:
        tuple: (vx, vy, vz) velocities in pixels/second
    """
    # Calculate velocity for each axis
    time = frame_diff / fps
    if time <= 0:
        return (0, 0, 0)
    
    vx = (pos2[0] - pos1[0]) / time
    vy = (pos2[1] - pos1[1]) / time
    vz = 0  # Ignore z for now
    
    return (vx, vy, vz)

def calculate_speed(pos1, pos2, frame_diff, fps=30):
    """Calculate overall speed (magnitude of velocity vector)"""
    vx, vy, vz = calculate_velocities(pos1, pos2, frame_diff, fps)
    return math.sqrt(vx**2 + vy**2 + vz**2)

def is_rolling(positions, frames, fps=30, window_size=5, min_y_velocity=30):
    """Determine if the ball is rolling based on vertical velocity.
    
    Args:
        positions: List of (x,y,z) positions
        frames: List of frame numbers
        fps: Frames per second
        window_size: Number of frames to analyze
        min_y_velocity: Minimum absolute Y velocity to consider ball not rolling
        
    Returns:
        bool: True if the ball appears to be rolling
    """
    if len(positions) < window_size + 1:
        return False
        
    # Calculate Y velocities for the window
    y_velocities = []
    for i in range(1, len(positions)):
        pos1 = positions[i-1]
        pos2 = positions[i]
        frame_diff = frames[i] - frames[i-1]
        _, vy, _ = calculate_velocities(pos1, pos2, frame_diff, fps)
        y_velocities.append(abs(vy))
    
    # Use the last window_size velocities
    recent_velocities = y_velocities[-window_size:]
    avg_y_velocity = np.mean(recent_velocities)
    
    # If vertical movement is very small, consider it rolling
    return avg_y_velocity < min_y_velocity

def filter_valid_tracks(tracks, min_speed=10, max_track_length=300, fps=30):
    valid_tracks = []
    for track in tracks:
        frames = [p[2] for p in track['positions']]
        positions = [(p[0], p[1], 0) for p in track['positions']]  # z=0 for now
        track_length = track['last_frame'] - track['start_frame']
        
        # Skip long tracks (likely spare balls)
        if track_length < fps / 1.2:
            print("bad", track_length, track['start_frame'], '-', track['last_frame'])
            continue

        # Calculate average speed
        speeds = []
        for i in range(1, len(track['positions'])):
            pos1 = track['positions'][i-1]
            pos2 = track['positions'][i]
            frame_diff = pos2[2] - pos1[2]
            speed = calculate_speed(pos1, pos2, frame_diff, fps)
            speeds.append(speed)
        
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Check if ball is rolling
        rolling = False # = is_rolling(positions, frames, fps)
        
        # Keep tracks with sufficient speed and not rolling
        if avg_speed > min_speed and not rolling:
            print("OK", avg_speed, track['start_frame'], '-', track['last_frame'], 'not rolling')
            valid_tracks.append(track)
        else:
            status = 'rolling' if rolling else 'slow'
            print("BAD", avg_speed, track['start_frame'], '-', track['last_frame'], status)
   
    return valid_tracks

def merge_tracks(tracks, max_gap_seconds=1, fps=60):
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

def crop_and_save_track(video_file, track, output_path):
    """
    video_file: путь к исходному видео
    track: словарь с ключами 'positions' (список (x, y, frame)), 'start_frame', 'last_frame'
    output_path: путь для сохранения результата
    crop_width: ширина crop (например 540)
    crop_height: если None, используется вся высота видео
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return
    crop_height = frame.shape[0]

    aspect_ratio = 9 / 16
    crop_width = int(crop_height * aspect_ratio)



    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    # Создаём словарь frame->(x, y) для быстрого доступа
    frame_to_pos = {int(pos[2]): (float(pos[0]), float(pos[1])) for pos in track['positions']}
    all_frames = sorted(frame_to_pos.keys())
    start_frame = track['start_frame']
    end_frame = track['last_frame']

    for frame_idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Усреднение положения мяча за последние 5 детекций
        recent_frames = [f for f in all_frames if f <= frame_idx]
        last_n_frames = recent_frames[-5:] if len(recent_frames) >= 1 else []
        if last_n_frames:
            xs = [frame_to_pos[f][0] for f in last_n_frames]
            x_center = int(np.mean(xs))
        else:
            x_center = frame.shape[1] // 2

        # Центр crop по X — усреднённая позиция мяча, по Y — по центру кадра (но crop всегда по всей высоте)
        y_center = frame.shape[0] // 2  # не используется, crop по всей высоте

        # Координаты crop
        x1 = max(0, x_center - crop_width // 2)
        x2 = x1 + crop_width
        if x2 > frame.shape[1]:
            x2 = frame.shape[1]
            x1 = x2 - crop_width
        y1 = 0
        y2 = y1 + crop_height
        if y2 > frame.shape[0]:
            y2 = frame.shape[0]
            y1 = 0

        # Crop и запись
        crop_frame = frame[y1:y2, x1:x2]
        cv2.imshow(f"Video: {video_file}", crop_frame)
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        out.write(crop_frame)

    cap.release()
    out.release()

def show_track_frames(video_file, tracks, preview_seconds=1, detect_json_dir=None, skip_no_track=True):
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
    sorted_tracks = sorted(tracks, key=lambda x: x['start_frame'])
    
    # Проходим по всем трекам
    # If skip_no_track is False, we show all frames in the video, overlaying track info if present.
    # If True, we show only frames with tracks (the original behavior, but now can be toggled).
    if not skip_no_track:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        track_idx = 0
        sorted_tracks = sorted(tracks, key=lambda x: x['start_frame'])
        while current_frame < total_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Find if current_frame is in any track
            in_track = False
            track = None
            for i, t in enumerate(sorted_tracks):
                if t['start_frame'] <= current_frame <= t['last_frame']:
                    in_track = True
                    track = t
                    track_num = i + 1
                    break
            # Overlay frame and track info
            cv2.putText(frame, f"Frame: {current_frame}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if in_track:
                cv2.putText(frame, f"Track: {track_num} ({track['start_frame']}-{track['last_frame']})", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            # Draw detection if detect_json_dir is not set
            if detect_json_dir is not None:
                import os
                json_path = os.path.join(detect_json_dir, f"frame_{current_frame}.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            det = json.load(jf)
                        # New format: list of dicts with cls_id==0 for ball
                        if isinstance(det, list):
                            for obj in det:
                                if isinstance(obj, dict) and obj.get('cls_id', -1) == 0:
                                    x1, y1, x2, y2 = int(obj['x1']), int(obj['y1']), int(obj['x2']), int(obj['y2'])
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    except Exception as e:
                        print(f"Ошибка чтения detect_json: {e}")
            cv2.imshow(f"Video: {video_file}", frame)
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                while True:
                    key = cv2.waitKey(0)
                    if key == ord(' '):
                        break
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                # No next track skipping in this mode
            current_frame += 1
        cap.release()
        cv2.destroyAllWindows()
        return
    # Original behavior: show only frames with tracks
    for i, track in enumerate(sorted_tracks):
        start_frame = max(0, track['start_frame'] - preview_frames)
        end_frame = track['last_frame'] - preview_frames
        print(f"\nТрек {i+1}: кадры {track['start_frame']}-{track['last_frame']}")
        print(f"Показываю кадры {start_frame}-{end_frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        while current_frame <= end_frame and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Overlay frame and track info
            cv2.putText(frame, f"Frame: {current_frame}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Track: {i+1} ({track['start_frame']}-{track['last_frame']})", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if track['start_frame'] <= current_frame <= track['last_frame']:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            # Draw detection if detect_json_dir is set
            if detect_json_dir is not None:
                import os
                json_path = os.path.join(detect_json_dir, f"frame_{current_frame}.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            det = json.load(jf)
                        # New format: list of dicts with cls_id==0 for ball
                        if isinstance(det, list):
                            for obj in det:
                                if isinstance(obj, dict) and obj.get('cls_id', -1) == 0:
                                    x1, y1, x2, y2 = int(obj['x1']), int(obj['y1']), int(obj['x2']), int(obj['y2'])
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    except Exception as e:
                        print(f"Ошибка чтения detect_json: {e}")
            cv2.imshow(f"Video: {video_file}", frame)
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                while True:
                    key = cv2.waitKey(0)
                    if key == ord(' '):
                        break
                    elif key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('n'):
                        break
                if key == ord('n'):
                    break
            elif key == ord('n'):
                break
            current_frame += 1
            if current_frame > end_frame:
                key = cv2.waitKey(2000)
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                break
    
    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


    
def main():
    # Load and process tracks
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("file_log", type=str, help="Path to the video file")
    parser.add_argument("video_file", type=str, help="Path to the video file")
    parser.add_argument("--detect_json_dir", type=str, default=None, help="Directory with detection JSON files (frame_{frame}.json)")
    parser.add_argument("--skip_no_track", type=lambda x: (str(x).lower() == 'true'), default=True, help="If True, only show frames with tracks. If False, show all frames.")
    args = parser.parse_args()

    file_log = args.file_log 
    video_file = args.video_file 
    detect_json_dir = args.detect_json_dir
    skip_no_track = args.skip_no_track

    #object_detector = cv2.createBackgroundSubtractorMOG2()

    tracks = load_tracks(file_log)
    valid_tracks = filter_valid_tracks(tracks)
    tracks_array = [{'start_frame': t['start_frame'], 'last_frame': t['last_frame']} for t in tracks]
    print('tracks', tracks_array)
    valid_tracks_array = [{'start_frame': t['start_frame'], 'last_frame': t['last_frame']} for t in valid_tracks]
    print('valid_tracks', valid_tracks_array)
    merged_tracks = merge_tracks(valid_tracks)
    valid_tracks_array = [{'start_frame': t['start_frame'], 'last_frame': t['last_frame']} for t in merged_tracks]
    print('merged_tracks', valid_tracks_array)
#    show_track_frames(video_file, merged_tracks, preview_seconds=0, detect_json_dir=detect_json_dir, skip_no_track=skip_no_track)
    ensure_reels_dir()
    for i, track in enumerate(merged_tracks):
        output_path = f"reels/reels_track_{i+1:06d}.mp4"
        crop_and_save_track(video_file, track, output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()