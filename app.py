from ultralytics import YOLO

# 1. Inizializza modello
model = YOLO('best.pt')  # Path al tuo modello

# 2. Esegui inferenza in batch sulla cartella dei frame
model.predict(
    source='frames',            # Cartella con tutti i frame estratti
    save=True,                  # Salva immagini annotate
    save_txt=True,              # <<<<<< SALVA I .TXT DELLE LABEL
    project='dataset_cvat',     # Cartella di output (puoi cambiarla)
    name='predict',             # Sottocartella di output (sempre la stessa)
    exist_ok=True               # Sovrascrive la cartella se già esiste
)

print("Inferenzione completata! Trovi immagini annotate e .txt in 'dataset_cvat/predict/'")
