import torch
import os
from ultralytics import YOLO
from roboflow import Roboflow
from dotenv import load_dotenv

# kullandığınız python versiyonu ile uyumunu kontrol edip uygun olan cudayı indirebilirsiniz
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# .env dosyasını yükleyin
load_dotenv()

# Roboflow API anahtarını al
api_key = os.getenv("ROBOFLOW_API_KEY")

# Roboflow ile bağlantı kurun ve veri setini indirin
rf = Roboflow(api_key=api_key)
project = rf.workspace("imageprocessing-rgrlh").project("apricot-yygxk")
version = project.version(1)
dataset = version.download("yolov8")
                

# Eğitim ve test işlemlerini ana kod bloğuna alıyoruz
# Modeli eğit
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)
