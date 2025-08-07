from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # il tuo modello

cap = cv2.VideoCapture("fedeschiuma.mp4")  # o 0 per webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()  # disegna le box

    cv2.imshow("YOLO Detection", annotated_frame)  # Mostra il frame

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
