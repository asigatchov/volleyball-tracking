import cv2
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Crop video based on log file.")
    parser.add_argument("--video", required=True, help="Path to the input video.")
    parser.add_argument("--log_txt", required=True, help="Path to the log file with object coordinates.")
    parser.add_argument("--output", required=False, default="output.mp4", help="Path to save the cropped video.")
    return parser.parse_args()

def read_log_file(log_path):
    detections = {}
    with open(log_path, "r") as file:
        for line in file:
            x, y, frame_num, radius = map(int, line.strip().split(";"))
            detections[frame_num] = (x, y, radius)
    return detections

def interpolate_missing_detections(detections, total_frames):
    interpolated = {}
    for frame in range(total_frames):
        if frame in detections:
            interpolated[frame] = detections[frame]
        else:
            prev_frame = max((f for f in detections if f < frame), default=None)
            next_frame = min((f for f in detections if f > frame), default=None)
            if prev_frame is not None and next_frame is not None:
                # Linear interpolation
                alpha = (frame - prev_frame) / (next_frame - prev_frame)
                prev_x, prev_y, _ = detections[prev_frame]
                next_x, next_y, _ = detections[next_frame]
                x = int(prev_x + alpha * (next_x - prev_x))
                y = int(prev_y + alpha * (next_y - prev_y))
                interpolated[frame] = (x, y, detections[prev_frame][2])  # Use radius from prev_frame
            elif prev_frame is not None:
                interpolated[frame] = detections[prev_frame]
            elif next_frame is not None:
                interpolated[frame] = detections[next_frame]
    return interpolated

def crop_video(video_path, detections, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    aspect_ratio = 9 / 16
    crop_width = int(frame_height * aspect_ratio)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, frame_height))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detections:
            x, _, _ = detections[frame_idx]
            left = max(0, x - crop_width // 2)
            right = min(frame_width, left + crop_width)
            left = right - crop_width  # Ensure exact crop width
            cropped_frame = frame[:, left:right]
            out.write(cropped_frame)

    cap.release()
    out.release()

def main():
    args = parse_arguments()
    detections = read_log_file(args.log_txt)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    detections = interpolate_missing_detections(detections, total_frames)
    crop_video(args.video, detections, args.output)

if __name__ == "__main__":
    main()



