# YOLO Object Detection - Real Time Video

Questo progetto utilizza [Ultralytics YOLO](https://docs.ultralytics.com) per eseguire **object detection in tempo reale** su un video (`video.mp4`) o sulla webcam, **senza salvare alcun file di output**.

---

## 📥 Come scaricare il progetto

Apri il terminale e clona il repository:

```bash
git clone https://github.com/Simone2408/YOLO.git
cd YOLO
```

---

## ⚙️ Requisiti

- **Python** 3.8 o superiore
- **Ultralytics YOLO**  
  Installa le dipendenze con:
  ```bash
  pip install ultralytics
  ```

Se vuoi sfruttare la **GPU**, assicurati di avere installato **PyTorch con supporto CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📦 Modello `best.pt`

- Se il file `best.pt` è già presente nella root del progetto, non devi fare nulla.
- Se non è presente o troppo grande per GitHub, scaricalo dal link fornito dal proprietario del progetto e posizionalo nella cartella principale (`YOLO/`).

Esempio:
```markdown
📥 Scarica `best.pt` da [LINK QUI] e mettilo nella cartella principale del progetto.
```

---

## ▶️ Uso di `test.py`

Per eseguire object detection **su un video** (annotato in tempo reale, senza salvare nulla):
```bash
python test.py
```

Per impostazione predefinita:
- Modello: `best.pt`
- Sorgente video: `video.mp4`
- Visualizzazione: sì (`show=True`)
- Salvataggio: no (`save=False`)

---

## 📷 Uso con la webcam

Per usare la webcam, apri `test.py` e modifica:
```python
source=0
```
oppure esegui:
```bash
python test.py --source 0
```
(se implementi la gestione argomenti).

---

## 🔧 Parametri utili (`test.py`)

- Cambiare soglia di confidenza:
  ```python
  conf=0.35
  ```
- Forzare CPU:
  ```python
  device="cpu"
  ```
- Forzare GPU 0:
  ```python
  device="0"
  ```

---

## 🛑 Come terminare
Premi **Q** oppure chiudi la finestra video per terminare l’esecuzione.

---

✏️ **Autore:** [Simone2408](https://github.com/Simone2408)  
📅 **Ultimo aggiornamento:** Agosto 2025
