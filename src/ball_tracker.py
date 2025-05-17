import numpy as np
from collections import deque
from scipy.spatial import distance
from dataclasses import dataclass, field
import json
from typing import List, Tuple, Dict, Optional, Any
import dataclasses

@dataclass
class Track:
    positions: deque = field(default_factory=lambda: deque(maxlen=1500))
    prediction: List[float] = field(default_factory=list)
    last_frame: int = 0
    start_frame: int = 0
    ball_size: float = 0  # Размер мяча в пикселях (диаметр)
    real_positions: deque = field(default_factory=lambda: deque(maxlen=1500))  # Позиции в реальных координатах (см)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Track to dictionary for JSON serialization"""
        return {
            'positions': list(self.positions),
            'real_positions': list(self.real_positions),
            'prediction': self.prediction,
            'last_frame': self.last_frame,
            'start_frame': self.start_frame,
            'ball_size': self.ball_size
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], buffer_size: int = 1500) -> 'Track':
        """Create Track from dictionary"""
        track = cls()
        track.positions = deque(data['positions'], maxlen=buffer_size)
        if 'real_positions' in data:
            track.real_positions = deque(data['real_positions'], maxlen=buffer_size)
        track.prediction = data['prediction']
        track.last_frame = data['last_frame']
        track.start_frame = data['start_frame']
        if 'ball_size' in data:
            track.ball_size = data['ball_size']
        return track

class BallTracker:
    def __init__(self, buffer_size=1500, max_disappeared=15, max_distance=100, ball_diameter_cm=21):
        self.next_id = 0
        self.tracks: Dict[int, Track] = {}
        self.buffer_size = buffer_size
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # в пикселях
        self.max_distance_cm = max_distance  # в сантиметрах
        self.ball_diameter_cm = ball_diameter_cm  # диаметр мяча в см
        
    def pixels_to_cm(self, pixels, ball_size):
        """Преобразование пикселей в сантиметры на основе размера мяча"""
        if ball_size <= 0:
            return pixels  # Если размер мяча неизвестен, возвращаем исходное значение
        scale = self.ball_diameter_cm / ball_size  # см/пиксель
        return pixels * scale
        
    def calculate_real_position(self, position, ball_size):
        """Рассчитывает реальные координаты в см на основе размера мяча"""
        if ball_size <= 0:
            return position  # Если размер мяча неизвестен, возвращаем исходное положение
        scale = self.ball_diameter_cm / ball_size  # см/пиксель
        return (position[0] * scale, position[1] * scale)
        
    def update(self, detections, frame_number):
        # Удаление треков, которые не обновлялись дольше max_disappeared кадров
        deleted_tracks = []
        for track_id in list(self.tracks.keys()):
            last_frame = self.tracks[track_id].last_frame
            if (frame_number - last_frame) > self.max_disappeared:
                deleted_tracks.append(self.tracks[track_id])
                del self.tracks[track_id]

        # Сопоставление детекций с существующими треками
        active_tracks = list(self.tracks.items())
        unused_detections = list(detections)
        
        # Проверяем формат детекций - старый или новый
        is_new_format = len(unused_detections) > 0 and isinstance(unused_detections[0], dict) and 'position' in unused_detections[0]
        
        # Матрица расстояний между всеми треками и детекциями
        distance_matrix = np.zeros((len(active_tracks), len(unused_detections)))
        for i, (track_id, track) in enumerate(active_tracks):
            if len(track.positions) > 0:
                last_pos, _ = track.positions[-1]
                last_pos = track.prediction
                for j, det in enumerate(unused_detections):
                    if is_new_format:
                        # Новый формат с информацией о размере мяча
                        det_pos = det['position']
                        distance_matrix[i, j] = distance.euclidean(last_pos, det_pos)
                    else:
                        # Старый формат - просто координаты
                        distance_matrix[i, j] = distance.euclidean(last_pos, det)

        # Жадное сопоставление по минимальному расстоянию
        matched_pairs = []
        used_detection_indices = set()
        while True:
            if distance_matrix.size == 0 or np.all(np.isinf(distance_matrix)):
                break
           
            min_val = np.min(distance_matrix)
            if min_val > self.max_distance:
                break
                
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            track_id, _ = active_tracks[i]
            det = unused_detections[j]
            
            if is_new_format:
                # Новый формат с информацией о размере мяча
                self._update_track(track_id, det['position'], frame_number, det['ball_size'])
            else:
                # Старый формат - просто координаты
                self._update_track(track_id, det, frame_number)
            matched_pairs.append((track_id, j))
            used_detection_indices.add(j)
            
            # Заполняем использованные значения большими числами
            distance_matrix[i, :] = np.inf
            distance_matrix[:, j] = np.inf

        # Добавление несопоставленных детекций как новые треки
        for j, det in enumerate(unused_detections):
            if j not in used_detection_indices:
                if is_new_format:
                    # Новый формат с информацией о размере мяча
                    self._add_track(det['position'], frame_number, det['ball_size'])
                else:
                    # Старый формат - просто координаты
                    self._add_track(det, frame_number)
                # print('add new track', det)

        return  self._get_main_ball(deleted_tracks) 

    def _add_track(self, position, frame_number, ball_size=0):
        track = Track()
        track.positions = deque([(position, frame_number)], maxlen=self.buffer_size)
        track.ball_size = ball_size
        
        # Рассчитываем реальные координаты в см
        real_position = self.calculate_real_position(position, ball_size)
        track.real_positions = deque([(real_position, frame_number)], maxlen=self.buffer_size)
        
        track.prediction = position
        track.last_frame = frame_number
        track.start_frame = frame_number
        
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _update_track(self, track_id, position, frame_number, ball_size=None):
        self.tracks[track_id].positions.append((position, frame_number))
        self.tracks[track_id].last_frame = frame_number
        
        # Обновляем размер мяча, если он предоставлен
        if ball_size is not None and ball_size > 0:
            self.tracks[track_id].ball_size = ball_size
        
        # Рассчитываем реальные координаты в см
        real_position = self.calculate_real_position(position, self.tracks[track_id].ball_size)
        self.tracks[track_id].real_positions.append((real_position, frame_number))
        
        # Обновление предсказания с учетом временного интервала
        if len(self.tracks[track_id].positions) > 1:
            prev_pos, prev_frame = self.tracks[track_id].positions[-2]
            dt = frame_number - prev_frame
            if dt == 0:
                dx = 0
                dy = 0
            else:
                dx = int((position[0] - prev_pos[0]) / dt)
                dy = int((position[1] - prev_pos[1]) / dt)
            self.tracks[track_id].prediction = [
                position[0] + dx,
                position[1] + dy
            ]
        else:
            self.tracks[track_id].prediction = position

    def _get_main_ball(self, deleted_tracks):
        main_ball = None
        max_score = -1
        
        for track_id, track in self.tracks.items():
            positions = [p for p, _ in track.positions]
            
            if len(positions) < 3:
                continue
                
            # Рассчитываем стабильность с учетом времени
            time_steps = [f for _, f in track.positions]
            velocities = []
            for i in range(1, len(positions)):
                dt = time_steps[i] - time_steps[i-1]
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                velocities.append((dx/dt, dy/dt))
                
            # Оценка стабильности скорости
            var = np.var(velocities, axis=0)
            stability = 1 / (np.sum(var) + 1e-5)
            
            # Дополнительный вес для более длинных треков
            length_weight = np.log(len(positions) + 1)
            
            total_score = stability * length_weight
            
            if total_score > max_score:
                max_score = total_score
                main_ball = track_id
        
        # Convert tracks to serializable format for returning
        tracks_dict = {track_id: track.to_dict() for track_id, track in self.tracks.items()}
                
        return main_ball, tracks_dict, deleted_tracks
        
    def to_json(self) -> str:
        """Serialize tracker state to JSON"""
        data = {
            'next_id': self.next_id,
            'tracks': {str(track_id): track.to_dict() for track_id, track in self.tracks.items()},
            'buffer_size': self.buffer_size,
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BallTracker':
        """Create tracker from JSON string"""
        data = json.loads(json_str)
        tracker = cls(
            buffer_size=data['buffer_size'],
            max_disappeared=data['max_disappeared'],
            max_distance=data['max_distance']
        )
        tracker.next_id = data['next_id']
        
        # Reconstruct tracks
        for track_id_str, track_data in data['tracks'].items():
            track_id = int(track_id_str)
            tracker.tracks[track_id] = Track.from_dict(track_data, buffer_size=tracker.buffer_size)
            
        return tracker