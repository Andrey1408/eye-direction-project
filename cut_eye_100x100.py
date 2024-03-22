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

def crop_eye_region(image, eye_points, target_size):
    # Извлечение координат для глаза
    x, y, w, h = cv2.boundingRect(np.array(eye_points))

    # Вычисление центра глаза
    eye_center = (x + w // 2, y + h // 2)

    # Вычисление координат для обрезки
    crop_x = max(0, eye_center[0] - target_size // 2)
    crop_y = max(0, eye_center[1] - target_size // 2)

    # Обрезка изображения
    cropped_eye = image[crop_y:crop_y + target_size, crop_x:crop_x + target_size]

    return cropped_eye

image_path = "test_img/img4.jpg"

# Предположим, что landmarks - это объект с координатами ключевых точек лица
# Ваш код для получения landmarks из results.multi_face_landmarks[0]
landmarks = find_face_landmarks(image_path)

# Загрузка изображения
image = cv2.imread(image_path)

# Индексы ключевых точек для левого и правого глаза (примерные значения, уточните их в соответствии с вашим форматом landmarks)
left_eye_indices = [362, 385, 387, 263, 373, 380]

# Извлечение координат для левого глаза
left_eye_points = [(int(landmarks[i].x * image.shape[1]), int(landmarks[i].y * image.shape[0])) for i in left_eye_indices]

# Обрезка области с глазом в центре
cropped_eye = crop_eye_region(image, left_eye_points, target_size=100)

# Отображение обрезанной области
cv2.imshow("Cropped Eye Region", cropped_eye)
cv2.waitKey(0)
cv2.destroyAllWindows()
