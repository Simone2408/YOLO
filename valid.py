from ultralytics import YOLO

model = YOLO('best.pt')
metrics = model.val(data='data.yaml')  # oppure il tuo file yaml
print(metrics)
