# YOLOv12 Thermal Detection 

Questa repo contiene un modello YOLOv12n allenato su immagini termiche e due script semplici per provarlo in locale.

---


## 📥 Come scaricare il progetto

```bash
git clone https://github.com/Simone2408/YOLO.git
cd YOLO
```
---


## 1. Preparazione ambiente

Consigliato usare un virtual environment per mantenere pulito Python:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate        # Windows PowerShell
```

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

---

## 2. File necessari


* `best.pt` → i pesi del modello YOLO allenato
* video termici, es. `video.mp4`
* gli script Python (`test_realtime.py`, `test_offline_save.py`)

---

## 3. Uso degli script

### 🔹 Test realtime (solo visualizzazione)

Apre il video e lo annota in tempo reale. Non salva nulla.

```bash
python3 test_realtime.py
```

### 🔹 Test offline (salvataggio output)

Elabora il video completo e salva un file annotato (`video_out.mp4`).

```bash
python3 test_offline_save.py
```

---

## 4. Note

* Funziona su CPU, quindi gira praticamente su qualsiasi PC.
*  `test_realtime.py` è ideale per mostrare le annotazioni in diretta, mentre `test_offline_save.py` fornisce un video annotato.

---

## 5. Licenza

licenza **MIT**.
Il modello YOLOv12 usato: [Ultralytics](https://github.com/ultralytics/ultralytics).

---
