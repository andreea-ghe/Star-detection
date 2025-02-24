from ultralytics import YOLO

def detection(image_path, model_path="runs/detect/constellation_run5/weights/best.pt"):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.25, save=False)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # bounding box coordinates
            conf = box.conf[0].item()      # confidence score
            cls_id = int(box.cls[0].item())  # class ID
            class_name = model.names[cls_id]
            
            detections.append({
                'class_id': cls_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })

    return detections
