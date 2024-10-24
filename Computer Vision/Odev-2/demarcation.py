import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Rastgele Arka Plan Rengi Oluşturma
height, width = 400, 400
background_color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # Rastgele renk
image = np.ones((height, width, 3), dtype=np.uint8) * background_color

# 2. Daire Çizme
center_coordinates = (200, 200)  # Dairenin merkezi
radius = 80  # Dairenin yarıçapı
color = (255, 255, 255)  # Dairenin rengi (beyaz)
thickness = -1  # Daireyi doldurmak için -1

cv2.circle(image, center_coordinates, radius, color, thickness)

# 3. Gri Tonlama
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Kabartma İşlemi (Laplacian)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# 5. Sonuçları Gösterme
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Gray Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Laplacian (Embossing)')
plt.imshow(laplacian, cmap='gray')
plt.axis('off')

plt.show()
