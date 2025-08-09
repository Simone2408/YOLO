# yolo_low_fps_mac_intel.py
# Usage:
#   python yolo_low_fps_mac_intel.py --source video.mp4
#   python yolo_low_fps_mac_intel.py --source 0          # webcam integrata

import argparse
import time
from pathlib import Path
import cv2

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser("YOLO low-FPS viewer (Mac Intel CPU)")
    p.add_argument("--weights", type=str, default="best.pt", help="Percorso al modello .pt")
    p.add_argument("--source", type=str, default="video.mp4",
                   help='File video o indice webcam (es. "0")')
    p.add_argument("--imgsz", type=int, default=512, help="Lato lungo per l'inferenza (più piccolo = meno carico)")
    p.add_argument("--conf", type=float, default=0.35, help="Soglia confidenza (più alta = meno box)")
    p.add_argument("--iou", type=float, default=0.45, help="Soglia IoU per NMS")
    p.add_argument("--vid_stride", type=int, default=4, help="Salta frame (1=nessuno, 4=1 su 4)")
    p.add_argument("--target_fps", type=float, default=10.0, help="FPS massimi da mostrare a schermo")
    p.add_argument("--max_det", type=int, default=100, help="Numero massimo di box per frame")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Filtra per classi (es. --classes 0 2); lascia vuoto per tutte")
    p.add_argument("--window", type=str, default="YOLO (ESC per uscire)", help="Nome finestra")
    p.add_argument("--resize_view", type=int, default=0,
                   help="Ridimensiona finestra (0=auto, >0 larghezza in px, altezza proporz.)")
    return p.parse_args()


def main():
    args = parse_args()

    # Sorgente: cast a int se è un numero (per webcam)
    src = args.source
    if src.isdigit():
        src = int(src)

    # Carica modello (CPU)
    model = YOLO(args.weights)
    try:
        # Piccolo speed-up fondendo conv+bn se applicabile
        model.fuse()
    except Exception:
        pass

    # Stream di inferenza (frame-by-frame) senza GUI interna
    results_gen = model.predict(
        source=src,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device="cpu",          # Mac Intel: CPU
        half=False,            # half solo su CUDA; su CPU lascia False
        stream=True,           # fondamentale per iterare
        vid_stride=args.vid_stride,
        max_det=args.max_det,
        classes=args.classes,
        agnostic_nms=False,
        show=False,
        save=False,
        verbose=False,
        workers=0              # evita overhead di dataloader multi-process
    )

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    # Se vogliamo un cap agli FPS a schermo
    frame_interval = 1.0 / max(args.target_fps, 0.0001)
    last_shown = 0.0

    # Semplice stima FPS di elaborazione per overlay
    ema_fps = None
    alpha = 0.1

    for r in results_gen:
        t0 = time.time()

        # Disegna annotazioni sul frame
        im = r.plot()

        # Ridimensionamento finestra, se richiesto
        if args.resize_view and args.resize_view > 0:
            h, w = im.shape[:2]
            new_w = args.resize_view
            new_h = int(h * (new_w / w))
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Aggiorna FPS stimati
        dt = max(time.time() - t0, 1e-6)
        inst_fps = 1.0 / dt
        ema_fps = inst_fps if ema_fps is None else (alpha * inst_fps + (1 - alpha) * ema_fps)

        # Overlay info utili
        txt = f"CPU Mac Intel | shown<= {args.target_fps:.1f} FPS | proc~ {ema_fps:.1f} FPS | stride {args.vid_stride} | imgsz {args.imgsz}"
        cv2.putText(im, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Rispetta il cap agli FPS mostrati (non blocca se l'elaborazione è più lenta)
        now = time.time()
        elapsed = now - last_shown
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_shown = time.time()

        cv2.imshow(args.window, im)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
            break

    cv2.destroyAllWindows()
    print("Finito. Chiudi la finestra del video per terminare.")


if __name__ == "__main__":
    main()
