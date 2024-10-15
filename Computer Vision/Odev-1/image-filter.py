import cv2
import numpy as np

image_path = "C:\Goruntu isleme\Odev-1\images.png"
original = cv2.imread(image_path)

blurred = cv2.blur(original, (5, 5))

laplace = cv2.Laplacian(blurred, cv2.CV_64F)
laplace = np.uint8(np.absolute(laplace))

desired_width = 500
desired_height = 400

original_resized = cv2.resize(original, (desired_width, desired_height))
blurred_resized = cv2.resize(blurred, (desired_width, desired_height))
laplace_resized = cv2.resize(laplace, (desired_width, desired_height))

result = np.hstack((original_resized, blurred_resized, laplace_resized))

cv2.imshow("Original | Blurred | Laplace", result)

cv2.waitKey(0)

cv2.destroyAllWindows()
