import cv2
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("runs/detect/train/weights/best.pt")  # Eğitilmiş modelin yolu

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
    annotated_frame = results[0].plot()  # Görüntüyü üzerinde tahmin sonuçları ile çizer

    # Sonucu göster
    cv2.imshow("Nesne Algılama - Test", annotated_frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
