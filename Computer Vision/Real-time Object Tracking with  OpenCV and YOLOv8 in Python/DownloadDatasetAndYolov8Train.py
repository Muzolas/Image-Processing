import os
from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO  # YOLOv8 eğitim fonksiyonlarını içerir

# .env dosyasını yükleyin
load_dotenv()

# Roboflow API anahtarını al
api_key = os.getenv("ROBOFLOW_API_KEY")

# Roboflow ile bağlantı kurun ve veri setini indirin
rf = Roboflow(api_key=api_key)
project = rf.workspace("yolo-practice-xgsmc").project("glasses-iy1og")
version = project.version(1)
dataset = version.download("yolov8")  # Veri YOLOv8 formatında indirildi

# YOLOv8 modelini eğitmek için yol
data_yaml_path = os.path.join(dataset.location, "data.yaml")  # İndirilen verinin konumu

# YOLOv8 modelini başlatın
model = YOLO("yolov8n.pt")  # YOLOv8 nano modelini kullanarak başlatıyoruz

# Eğitimi başlatın
model.train(data=data_yaml_path, epochs=100, imgsz=640)
