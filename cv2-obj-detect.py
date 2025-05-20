import cv2

import numpy as np

import argparse


from src.ball_tracker import BallTracker, Track

# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")

parser.add_argument("video_file", type=str, help="Path to the video file")
args = parser.parse_args()

# Параметры для фильтрации окружностей
MIN_RADIUS = 10      # Минимальный радиус окружности
MAX_RADIUS = 100     # Максимальный радиус окружности
MIN_CIRCULARITY = 0.5 # Минимальная "круглость" (1 - идеальная окружность)
WHITE_THRESH = 200   # Порог для белого цвета


video_file = args.video_file  


object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=30)


tracker = BallTracker(buffer_size=1500)


def show_video(video_file):

    cap = cv2.VideoCapture(video_file) # file name


    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    frame_num = 0
    while cap.isOpened():
        frame_num +=1
        success, img = cap.read()
        if not success:
            break
        
        mask = object_detector.apply(img)
        
        # Морфологические операции для улучшения маски
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        _, mask = cv2.threshold(mask, 253, 255, cv2.THRESH_BINARY)
        # Поиск контуров
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
        
        #contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        boxs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
            if circularity < MIN_CIRCULARITY:
                #print('cir', circularity)
                continue
       
            print('cnt',cnt)
            if 50 < area  and area < 600:
                cv2.drawContours(img, [cnt], -1 , (0,255,0))
                x,y, w,h = cv2.boundingRect(cnt)
                aspect = h / w
                print('aspect', aspect)
                if 0.8  < aspect and aspect < 1.5:
                    box = {'x1': x, 'y1': y , 'x2': x+w, 'y2': y + h}

                    boxs.append(box)
                    
                    cv2.rectangle(img, (x,y), (x+w,y+ h), (0,255,0 ),3)
                    label = f"{w}x{h}"
                    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    #cv2.rectangle(mask, (x,y), (x+w,y+ h), (0,255,0 ),3)

        main_id, tracks, deleted_tracks = tracker.update(boxs,frame_num)



        for track_id, track_data in tracks.items():
            color = (255, 255, 0) if int(track_id) == main_id else (0, 0, 255)
            positions = list(track_data['positions'])

            if len(positions) < 3:
                continue

            print('track', track_data)
            # Рисуем трек
            for i in range(1, len(positions)):

                cv2.line(img,
                        (int(positions[i-1][0][0]), int(positions[i-1][0][1])),
                        (int(positions[i][0][0]), int(positions[i][0][1])),
                        color, 2)

            # Рисуем текущую позицию
            x, y = int(positions[-1][0][0]), int(positions[-1][0][1])
            
            f_diff = track_data['last_frame'] - track_data['start_frame']
            t_time = f_diff / fps
            cv2.putText(img, f'id: {track_id}: {t_time:.2f} {f_diff}', (x+25, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.circle(img,
                    (x,y),
                    10, color, -1)

        combined = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

        # Или вертикально (одно под другим)
        # combined = np.vstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        
        cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)  # Создаем окно с возможностью изменения размера
        cv2.imshow('Combined View', combined)
        
        #cv2.imshow('Frame', img)

        #cv2.imshow('mask', mask)

        key = cv2.waitKey(0)
        if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        


show_video(video_file)


