import numpy as np
from collections import deque
from scipy.spatial import distance

class BallTracker:
    def __init__(self, buffer_size=15, max_disappeared=5, max_distance=150):
        self.next_id = 0
        self.tracks = {}
        self.buffer_size = buffer_size
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def update(self, detections, frame_number):
        # Удаление треков, которые не обновлялись дольше max_disappeared кадров
        for track_id in list(self.tracks.keys()):
            last_frame = self.tracks[track_id]['last_frame']
            if (frame_number - last_frame) > self.max_disappeared:
                del self.tracks[track_id]

        # Сопоставление детекций с существующими треками
        active_tracks = list(self.tracks.items())
        unused_detections = list(detections)
        
        # Матрица расстояний между всеми треками и детекциями
        distance_matrix = np.zeros((len(active_tracks), len(unused_detections)))
        for i, (track_id, track) in enumerate(active_tracks):
            last_pos, _ = track['positions'][-1]
            for j, det in enumerate(unused_detections):
                distance_matrix[i, j] = distance.euclidean(last_pos, det)

        # Жадное сопоставление по минимальному расстоянию
        matched_pairs = []
        while True:
            if distance_matrix.size == 0 or np.all(np.isinf(distance_matrix)):
                break
           
            min_val = np.min(distance_matrix)
            if min_val > self.max_distance:
                break
                
            i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
            track_id, _ = active_tracks[i]
            det = unused_detections[j]
            
            self._update_track(track_id, det, frame_number)
            matched_pairs.append((track_id, det))
            
            # Заполняем использованные значения большими числами
            distance_matrix[i, :] = np.inf
            distance_matrix[:, j] = np.inf

        # Добавление несопоставленных детекций как новые треки
        for j, det in enumerate(unused_detections):
            if all(j != pair[1] for pair in matched_pairs):
                self._add_track(det, frame_number)

        return self._get_main_ball()

    def _add_track(self, position, frame_number):
        self.tracks[self.next_id] = {
            'positions': deque([(position, frame_number)], maxlen=self.buffer_size),
            'prediction': position,
            'last_frame': frame_number,
            'start_frame': frame_number
        }
        self.next_id += 1

    def _update_track(self, track_id, position, frame_number):
        self.tracks[track_id]['positions'].append((position, frame_number))
        self.tracks[track_id]['last_frame'] = frame_number
        
        # Обновление предсказания с учетом временного интервала
        if len(self.tracks[track_id]['positions']) > 1:
            prev_pos, prev_frame = self.tracks[track_id]['positions'][-2]
            dt = frame_number - prev_frame
            dx = (position[0] - prev_pos[0]) / dt
            dy = (position[1] - prev_pos[1]) / dt
            self.tracks[track_id]['prediction'] = [
                position[0] + dx * (frame_number - prev_frame),
                position[1] + dy * (frame_number - prev_frame)
            ]
        else:
            self.tracks[track_id]['prediction'] = position

    def _get_main_ball(self):
        main_ball = None
        max_score = -1
        
        for track_id, track in self.tracks.items():
            positions = [p for p, _ in track['positions']]
            
            if len(positions) < 3:
                continue
                
            # Рассчитываем стабильность с учетом времени
            time_steps = [f for _, f in track['positions']]
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
                
        return main_ball, self.tracks