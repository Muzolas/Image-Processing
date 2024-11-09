import cv2
from ultralytics import YOLO

# Eğitilmiş modeli yükle
model = YOLO("runs/detect/train/weights/best.pt")  # Eğitilmiş modelin yolunu buraya yazın

# Video dosyasını açın
video_path = "test_video.mp4"  # Test etmek istediğiniz videonun yolunu buraya yazın
cap = cv2.VideoCapture(video_path)

# Video dosyasının kare boyutlarını ve FPS değerini alarak çıktı videosu oluşturma
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(
    "output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video bitti veya okunamadı.")
        break

    # Model ile tahmin yap
    results = model(frame)

    # Tahmin sonuçlarını görüntü üzerine çiz
    annotated_frame = results[0].plot()

    # Çerçeveyi ekranda göster
    cv2.imshow("Nesne Algılama - Video", annotated_frame)

    # Çıktı videosuna yaz
    output_video.write(annotated_frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
output_video.release()
cv2.destroyAllWindows()
