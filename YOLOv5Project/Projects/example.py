import os
from pathlib import Path
import torch

# YOLOv5'i çalıştıracak fonksiyon
def run_yolo():
    # Modelleri, kaynakları ve diğer ayarları belirleme
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 küçük modelini yükler
    source = 'C:/Users/Muzolas/Desktop/a.jpeg'  # Algılama yapılacak resim dosyasının yolu

    # Algılama işlemi
    results = model(source)

    # Sonuçları kaydet
    results.save()  # Varsayılan olarak runs/detect klasörüne kaydeder
    results.show()  # Sonuçları ekranda gösterir

if __name__ == "__main__":
    run_yolo()
