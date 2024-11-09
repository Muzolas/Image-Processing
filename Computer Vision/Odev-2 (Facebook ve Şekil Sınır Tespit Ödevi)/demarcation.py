import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
image_path = 'C:\\Image Processing\\Computer Vision\\Odev-2\\circle.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Görüntü yüklenemedi. Lütfen yolun doğru olduğundan emin olun: {image_path}")
    exit()

if image.size == 0:
    print("Yüklenen görüntü boş. Lütfen geçerli bir görüntü dosyası kullanın.")
    exit()

# X ve Y yönündeki gradyanları Sobel operatörü ile hesapla
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3) 

# Gradyan büyüklüğünü hesapla
magnitude = np.sqrt(grad_x**2 + grad_y**2)

# Gradyan yönünü hesapla
angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)

# Gradyan büyüklüğünü ve yönünü normalize et
magnitude_normalized = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
angle_normalized = ((angle + 180) / 360 * 255).astype(np.uint8)  # 0-255 aralığına normalize et

# Gradyan büyüklüğünü ve yönünü görselleştir ve kaydet
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Gradyan Büyüklüğü")
plt.imshow(magnitude_normalized, cmap='gray')
plt.axis('off') 
plt.savefig('gradyan_buyuklugu.png') 

plt.subplot(1, 2, 2)
plt.title("Gradyan Yönü")
plt.imshow(angle_normalized, cmap='gray') 
plt.axis('off')  
plt.savefig('gradyan_yonu.png') 

plt.tight_layout()  
plt.show()

# Gradyan büyüklüğü ve yönünü ayrı ayrı kaydetme
cv2.imwrite('C:\\Image Processing\\Computer Vision\\Odev-2\\gradyan_buyuklugu_cv2.png', magnitude_normalized)
cv2.imwrite('C:\\Image Processing\\Computer Vision\\Odev-2\\gradyan_yonu_cv2.png', angle_normalized)

print("Gradyan büyüklüğü ve yönü başarıyla kaydedildi.")
