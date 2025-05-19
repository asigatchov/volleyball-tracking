
import cv2


import argparse




# Добавление парсинга аргументов командной строки
parser = argparse.ArgumentParser(description="Process a video file.")

parser.add_argument("video_file", type=str, help="Path to the video file")
args = parser.parse_args()

video_file = args.video_file  


object_detector = cv2.createBackgroundSubtractorMOG2(    history=100, varThreshold=30)




def show_video(video_file):

    cap = cv2.VideoCapture(video_file) # file name
    
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        mask = object_detector.apply(img)
        #_, mask = cv2.threshold(mask, 122, 255, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100 and area < 500:
                #cv2.drawContours(img, [cnt], -1 , (0,255,0))
                x,y, w,h = cv2.boundingRect(cnt)
                aspect = w / h

                if 0.9  < aspect and aspect < 1.1:
                    cv2.rectangle(img, (x,y), (x+w,y+ h), (0,255,0 ),3)
                    cv2.rectangle(mask, (x,y), (x+w,y+ h), (0,255,0 ),3)


        cv2.imshow('Frame', img)

        cv2.imshow('mask', mask)

        key = cv2.waitKey(0)
        if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        


show_video(video_file)


