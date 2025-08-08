# test.py
from ultralytics import YOLO

# Carica il modello
model = YOLO("best.pt")  # metti qui il tuo .pt

# Mostra il video annotato in tempo reale (nessun file salvato)
model.predict(
    source="video.mp4",  # oppure 0 per webcam
    conf=0.25,           # soglia confidenza (regolabile)
    show=True,           # apre la finestra con le annotazioni
    save=False,          # NON salva nulla
    device=""            # "" auto, "cpu" o "0" per GPU 0
)

print("Finito. Chiudi la finestra del video per terminare.")
