import cv2

# Загрузка изображения
image = cv2.imread("test_img/left_eye2.jpg", cv2.IMREAD_GRAYSCALE)

# Список функций обработки
threshold_methods = [
    cv2.THRESH_BINARY,
    cv2.THRESH_BINARY_INV,
    cv2.THRESH_TRUNC,
    cv2.THRESH_TOZERO,
    cv2.THRESH_TOZERO_INV
]

# Эксперимент с различными значениями порога и функциями обработки
for threshold_method in threshold_methods:
    for threshold_value in range(0, 255, 10):
        _, thresholded_image = cv2.threshold(image, threshold_value, 255, threshold_method)
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circle_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Преобразование в трехканальное изображение

        for contour in contours:
            # Вычисление моментов контура
            M = cv2.moments(contour)
            
            # Проверка на ненулевую площадь контура перед выделением центра
            if M['m00'] != 0:
                # Вычисление центра контура
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                
                # Рисование круга в центре контура
                cv2.circle(circle_image, (cx, cy), 3, (0, 255, 0), -1)
        
        # Проверка наличия ровно 4 центров
        if len(contours) == 1:
            # Визуализация результатов
            cv2.imshow(f'Threshold Method = {threshold_method}, Threshold Value = {threshold_value}', circle_image)
            cv2.waitKey(1000)  # Задержка 0.5 секунды
            cv2.destroyAllWindows()

# Завершение программы
cv2.destroyAllWindows()

# THRESH_BINARY_INV THRESH_BINARY 140
