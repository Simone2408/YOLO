import cv2, time, numpy as np
from ultralytics import YOLO

# === CONFIG  ===
MODEL_PATH = "best.pt"    # pesi del tuo modello
VIDEO_IN   = "video.mp4"  # file video (oppure "0" per webcam)
IMGSZ      = 640          # 704/768 per oggetti piccoli 
CONF       = 0.45         # soglia confidenza
IOU        = 0.40         # NMS più aggressiva = meno doppioni


def thermal_to_3ch(frame: np.ndarray) -> np.ndarray:
    """Converte frame termico 16/8-bit mono in 8-bit BGR 3 canali ."""
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
    frame_interval = 1.0 / fps
    w, h = int(cap.get(3)), int(cap.get(4))

    win = "YOLOv12 (CPU) - ESC per uscire"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))

    try:
        while True:
            t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok: break

            frame = thermal_to_3ch(frame)

            res = model.predict(frame, device="cpu", imgsz=IMGSZ,
                                conf=CONF, iou=IOU, agnostic_nms=True,
                                verbose=False)
            annotated = res[0].plot()

            cv2.imshow(win, annotated)
            if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                break

            # mantieni la velocità originale del video
            elapsed = time.perf_counter() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
