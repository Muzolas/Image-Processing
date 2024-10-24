import cv2
import torch

# Modeli yükle (yolov5s, yolov5m, yolov5l, yolov5x)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir görüntü al
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Model ile görüntü üzerinde tahmin yap
    results = model(frame)

    # Tahmin sonuçlarını görüntüle
    results.render()  # Çizim yap

    # Sonucu göster
    cv2.imshow('Nesne Algılama', results.ims[0])  # 'imgs' yerine 'ims' kullanıldı

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
