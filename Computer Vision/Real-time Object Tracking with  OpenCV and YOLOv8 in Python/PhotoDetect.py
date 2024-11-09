from ultralytics import YOLO
import cv2

# Eğitilmiş modeli yükle
model = YOLO("runs/detect/train/weights/best.pt")

# Test için görüntüyü yükleyin
img_path = "test_image.jpg"  # Test edeceğiniz görüntünün yolu
img = cv2.imread(img_path)

# Model ile tahmin yap
results = model(img)

# Tahmin sonuçlarını görüntüle
annotated_img = results[0].plot()

# Sonucu göster
cv2.imshow("Gözlük Algılama - Test Görüntüsü", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
