import cv2
import numpy as np
import mediapipe as mp

def find_face_landmarks(image_path):
    # Создание объекта MediaPipe Face
    face_mesh = mp.solutions.face_mesh.FaceMesh()

    # Загрузка изображения
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Конвертация изображения в RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обнаружение ключевых точек лица
    results = face_mesh.process(rgb_image)
    landmarks = results.multi_face_landmarks

    # Если найдены ключевые точки, возвращаем их координаты
    if landmarks:
        print("Face landmarks found:", landmarks)
        return landmarks[0].landmark

    print("No face landmarks found.")
    return None

image_path = "test_img/img4.jpg"

# Предположим, что landmarks - это объект с координатами ключевых точек лица
# Ваш код для получения landmarks из results.multi_face_landmarks[0]
landmarks = find_face_landmarks(image_path)

# Загрузка изображения
image = cv2.imread(image_path)

# Индексы ключевых точек для левого и правого глаза (примерные значения, уточните их в соответствии с вашим форматом landmarks)
left_eye_indices = [362, 385, 387, 263, 373, 380]
right_eye_indices = [33, 160, 158, 133, 153, 144]

# Извлечение координат для левого и правого глаза
left_eye_points = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in left_eye_indices]
right_eye_points = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in right_eye_indices]

# Выделение и отображение левого глаза
x_left, y_left, w_left, h_left = cv2.boundingRect(np.array(left_eye_points))
left_eye_image = image[y_left:y_left + h_left, x_left:x_left + w_left]
cv2.imshow("Left Eye", left_eye_image)
cv2.imwrite("test_img/left_eye2.jpg", left_eye_image)
cv2.waitKey(0)

# Выделение и отображение правого глаза
x_right, y_right, w_right, h_right = cv2.boundingRect(np.array(right_eye_points))
right_eye_image = image[y_right:y_right + h_right, x_right:x_right + w_right]
# cv2.imshow("Right Eye", right_eye_image)
# cv2.imwrite("test_img/right_eye.jpg", right_eye_image)
cv2.waitKey(0)

# Закрытие всех окон после завершения
cv2.destroyAllWindows()



