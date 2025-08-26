
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


# 1. Load model
model = YOLO("best.pt")

# 2. Run validation
results = model.val(data="data.yaml", split="val", save=True, save_json=True)


#   Class-level metrics

metrics = results.results_dict
names = results.names  # class names from YOLO

print("\n--- Per-Class Metrics ---")
for i, cls_name in names.items():
    p = metrics['metrics/precision(B)'][i]
    r = metrics['metrics/recall(B)'][i]
    map50 = metrics['metrics/mAP50(B)'][i]
    print(f"{cls_name}: Precision={p:.3f}, Recall={r:.3f}, mAP50={map50:.3f}")

#   Confusion Matrix
# Extract predictions and labels

y_true, y_pred = [], []
for pred in results.pred:
    if pred.boxes is None:  # skip empty
        continue
    labels = pred.boxes.cls.cpu().numpy().astype(int)
    # ground truth (YOLO provides in pred path)
    gt = pred.boxes.gt_cls.cpu().numpy().astype(int) if hasattr(pred.boxes, "gt_cls") else []
    y_true.extend(gt)
    y_pred.extend(labels)

if len(y_true) > 0:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(names.values()))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    plt.title("Confusion Matrix - YOLOv12n (Thermal Dataset)")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
else:
    print("⚠️ Could not compute confusion matrix (no GT labels available in results).")

print("\n--- Failure cases saved in runs/val/ ---")
print("Check runs/val/exp*/ for predicted images with errors (FP/FN).")
print("You can manually pick 5–10 bad examples for the thesis.")
