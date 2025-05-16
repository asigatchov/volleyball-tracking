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
        # Удаление старых треков
        for track_id in list(self.tracks.keys()):
            
            print('update', track_id, '--', self.tracks[track_id]['disappeared'])
            if self.tracks[track_id]['disappeared'] >= self.max_disappeared:
                import pdb; pdb.set_trace()
                del self.tracks[track_id]
                

        # Инициализация новых треков
        if not self.tracks and detections:
            for det in detections:
                self._add_track(det, frame_number)
        else:
            # Сопоставление детекций с существующими треками
            distances = {}
            for track_id, track in self.tracks.items():
                last_pos, _ = track['positions'][-1]
                for det in detections:
                    dist = distance.euclidean(last_pos, det)
                    distances[(track_id, tuple(det))] = dist

            # Жадное сопоставление по минимальному расстоянию
            matched = set()
            for (track_id, det), dist in sorted(distances.items(), key=lambda x: x[1]):
                if dist > self.max_distance:
                    continue
                if track_id not in matched and tuple(det) not in matched:
                    self._update_track(track_id, det, frame_number)
                    matched.add(track_id)
                    matched.add(tuple(det))

            # Добавление несопоставленных детекций как новые треки
            for det in detections:
                if tuple(det) not in matched:
                    self._add_track(det, frame_number)

        return self._get_main_ball()

    def _add_track(self, position, frame_number):
        self.tracks[self.next_id] = {
            'positions': deque([(position, frame_number)], maxlen=self.buffer_size),
            'disappeared': 0,
            'prediction': position
        }
        self.next_id += 1

    def _update_track(self, track_id, position, frame_number):
        self.tracks[track_id]['positions'].append((position, frame_number))
        self.tracks[track_id]['disappeared'] = 0
        # Простое предсказание следующей позиции
        if len(self.tracks[track_id]['positions']) > 1:
            prev_pos, _ = self.tracks[track_id]['positions'][-2]
            dx = position[0] - prev_pos[0]
            dy = position[1] - prev_pos[1]
            self.tracks[track_id]['prediction'] = [position[0] + dx, position[1] + dy]
        else:
            self.tracks[track_id]['prediction'] = position

    def _get_main_ball(self):
        # Выбор главного мяча по самой длинной и стабильной траектории
        main_ball = None
        max_score = -1
        
        for track_id, track in self.tracks.items():
            if len(track['positions']) < 3:
                continue
                
            # Расчет стабильности траектории
            dx = np.std([track['positions'][i][0][0] - track['positions'][i-1][0][0] 
                       for i in range(1, len(track['positions']))])
            dy = np.std([track['positions'][i][0][1] - track['positions'][i-1][0][1] 
                       for i in range(1, len(track['positions']))])
            stability = 1 / (dx + dy + 1e-5)
            
            if stability > max_score:
                max_score = stability
                main_ball = track_id
                print('get_main_ball:', max_score, main_ball)
                
        return main_ball, self.tracks
