import cv2


def create_video_writer(video_cap, output_filename, width = None , height= None):
    if None in [width, height]:
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        frame_width = width
        frame_height = height

        
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,(frame_width, frame_height))
    return writer

