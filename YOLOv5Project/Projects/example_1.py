from monai.data.ultrasound_confidence_map import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

source = "https://static.ticimax.cloud/cdn-cgi/image/width=861,quality=99/56107/uploads/blog/buyumeyen-kopek-cinsleri-ve-ozellikleri-9a2e.jpg"

results= model(source)

for result in results:
    annotated_img = result.plot()  # plot() fonksiyonuyla sonucun çizilmesi

    # Görüntüyü OpenCV ile göster
    cv2.imshow('YOLOv8 Detection', annotated_img)
    cv2.waitKey(0)  # Bir tuşa basılana kadar bekle