import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread("test_img/left_eye2.jpg")

# Преобразование в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применение пороговой обработки для выделения темных областей
_, binary_image = cv2.threshold(gray_image, 140, 250, cv2.THRESH_BINARY_INV)

# Нахождение контуров на исходном изображении
contours_original, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours_original = image.copy()
cv2.drawContours(image_with_contours_original, contours_original, -1, (0, 255, 0), 2)

# Определение ROI (области интереса), исключая темные края
x, y, w, h = cv2.boundingRect(np.vstack(contours_original))

# Вычисление значения обрезки (10% от максимального размера)
crop_percentage = 0.1
crop_size = int(max(w, h) * crop_percentage)

# Обрезка изображения, исключая темные области
cropped_image = image[y + crop_size:y + h - crop_size, x + crop_size:x + w - crop_size]

# Преобразование обрезанного изображения в оттенки серого
gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Применение пороговой обработки для выделения контуров на обрезанном изображении
_, binary_cropped_image = cv2.threshold(gray_cropped_image, 140, 250, cv2.THRESH_BINARY_INV)

# Нахождение контуров на обрезанном изображении
contours_cropped, _ = cv2.findContours(binary_cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours_cropped = cropped_image.copy()
cv2.drawContours(image_with_contours_cropped, contours_cropped, -1, (0, 255, 0), 2)

# Визуализация результатов
cv2.imshow('Original Image with Contours', image_with_contours_original)
cv2.imshow('Cropped Image with Contours', image_with_contours_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
