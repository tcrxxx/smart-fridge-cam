import cv2
from ultralytics import YOLO

# Load YOLOv8 model
# yolov8n.pt (nano), 
# yolov8s.pt (small), 
# yolov8m.pt (medium), 
# yolov8l.pt (large) or 
# yolov8x.pt (extra-large).
model = YOLO('yolov8m.pt')


# Cam init
cap = cv2.VideoCapture(0)  # Use '0' to integrated webcam, or adjust

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict Objects
    results = model(frame, conf_thres=0.5, iou_thres=0.4)

    # Extract Detections
    detections = results[0]
    
    # Draw bounding box and label
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinator
        label = model.names[int(box.cls[0])]    # Classe Name
        confidence = box.conf[0].item()         # Confiance
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame with detections
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press key 'q' to end loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Cam and close windows
cap.release()
cv2.destroyAllWindows()
