import cv2, numpy as np
from ultralytics import YOLO

# === CONFIG  ===
MODEL_PATH = "best.pt"
VIDEO_IN   = "video2.mp4"       
VIDEO_OUT  = "video_out.mp4"
IMGSZ      = 640
CONF       = 0.45
IOU        = 0.40
# ================================

def thermal_to_3ch(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        return frame
    if frame.dtype != np.uint8:
        mn, mx = float(frame.min()), float(frame.max())
        frame = ((frame - mn) / (mx - mn) * 255.0).clip(0,255).astype(np.uint8) if mx > mn else frame.astype(np.uint8)
    if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame

def main():
    model = YOLO(MODEL_PATH)
    try: model.fuse()
    except: pass

    src = int(VIDEO_IN) if isinstance(VIDEO_IN, str) and VIDEO_IN.isdigit() else VIDEO_IN
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Impossibile aprire: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break

            frame = thermal_to_3ch(frame)
            res = model.predict(frame, device="cpu", imgsz=IMGSZ,
                                conf=CONF, iou=IOU, agnostic_nms=True,
                                verbose=False)
            annotated = res[0].plot()
            out.write(annotated)

    finally:
        cap.release()
        out.release()

    print(f"Salvato: {VIDEO_OUT}")

if __name__ == "__main__":
    main()
