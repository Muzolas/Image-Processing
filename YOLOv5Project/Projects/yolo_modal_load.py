import torch
from pathlib import Path

# YOLOv5 modelini yükle
def load_yolo_model():
    # Yolov5s modelini ağırlıklarıyla beraber yükle
    model_path = 'C:/Goruntu isleme/yolov5s.pt'  # yolov5s ağırlıkları dosyasının yolu
    model = torch.load(model_path)  # Modeli yükler

    # Model özetini ekrana yazdır
    print(model)

    return model

if __name__ == "__main__":
    # Modeli yükleyip çıktısını al
    model = load_yolo_model()
