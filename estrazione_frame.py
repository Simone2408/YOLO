import cv2
import os
from glob import glob

video_folder = 'video'
output_folder = 'frames'

os.makedirs(output_folder, exist_ok=True)
video_files = glob(os.path.join(video_folder, '*.mp4'))

frame_interval = 3yol  # secondi
global_count = 0    # Contatore globale per tutti i frame

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Non posso aprire {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    sec = 0

    while sec < duration:
        frame_id = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        out_path = os.path.join(output_folder, f"frame{global_count:05d}.jpg")
        cv2.imwrite(out_path, frame)
        global_count += 1
        sec += frame_interval

    cap.release()
    print(f"Frame estratti da {video_path}")

print(f"Estrazione completata! {global_count} frame totali.")
