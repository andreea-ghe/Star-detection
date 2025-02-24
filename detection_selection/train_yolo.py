from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data="C:/Users/Andreea/Documents/dataset/second/data.yaml",
        epochs=20, 
        imgsz=640, 
        name='constellation_run'
    )

    print("Training completed and model saved.")

if __name__ == "__main__":
    train_model()
