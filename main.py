import cv2
from ultralytics import YOLO

# Carregue o modelo YOLOv8
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' para o modelo nano, ou ajuste conforme necessário

# Inicialize a câmera
cap = cv2.VideoCapture(0)  # Use '0' para a webcam integrada, ou ajuste conforme necessário

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realize a detecção de objetos
    results = model(frame)

    # Desenhe as caixas delimitadoras e etiquetas no frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            label = box.cls_name
            confidence = box.conf
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostre o frame com as detecções
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Saia do loop ao pressionar a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a câmera e feche as janelas
cap.release()
cv2.destroyAllWindows()
