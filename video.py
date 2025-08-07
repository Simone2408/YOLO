import cv2
import os
import re

# -------- Config --------
frames_dir = "dataset_cvat/predict"   # cartella con i frame annotati
output_video = "annotated_sequence.mp4"
fps = 10  # cambia a piacere (es. 5, 10, 15, 25...)

# Estensioni immagini accettate
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

def natural_key(s: str):
    """Ordina in modo 'naturale': frame_2 < frame_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

# Raccogli e ordina i frame
files = [f for f in os.listdir(frames_dir) if f.lower().endswith(IMG_EXTS)]
files.sort(key=natural_key)

if not files:
    raise SystemExit(f"Nessuna immagine trovata in '{frames_dir}' con estensioni {IMG_EXTS}")

# Legge il primo frame per ricavare dimensioni video
first_path = os.path.join(frames_dir, files[0])
first = cv2.imread(first_path)
if first is None:
    raise SystemExit(f"Impossibile leggere il primo frame: {first_path}")
h, w = first.shape[:2]

# Inizializza writer video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # compatibile con .mp4
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

skipped = 0
for name in files:
    fpath = os.path.join(frames_dir, name)
    frame = cv2.imread(fpath)
    if frame is None:
        print(f"[WARN] Immagine non leggibile, salto: {fpath}")
        skipped += 1
        continue
    # Se per qualche motivo dimensioni diverse, ridimensiona al formato del video
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    out.write(frame)

out.release()
print(f"✅ Video creato: {output_video}")
if skipped:
    print(f"ℹ️ Frame saltati perché non leggibili: {skipped}")
