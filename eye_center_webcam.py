import cv2
import numpy as np
import mediapipe as mp

# Создание объекта VideoCapture для веб-камеры (обычно 0 для встроенной веб-камеры)
cap = cv2.VideoCapture(0)

def find_face_landmarks(frame):
    # Создание объекта MediaPipe Face
    face_mesh = mp.solutions.face_mesh.FaceMesh()

    # Загрузка изображения
    height, width, _ = frame.shape

    # Конвертация изображения в RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обнаружение ключевых точек лица
    results = face_mesh.process(rgb_image)
    landmarks = results.multi_face_landmarks

    # Если найдены ключевые точки, возвращаем их координаты
    if landmarks:
        print("Face landmarks found:", landmarks)
        return landmarks[0].landmark

    print("No face landmarks found.")
    return None

while True:
    # Считывание кадра с веб-камеры
    ret, frame = cap.read()

    # Нахождение ключевых точек лица
    landmarks = find_face_landmarks(frame)

    if landmarks:
        # Индексы ключевых точек для левого и правого глаза (примерные значения, уточните их в соответствии с вашим форматом landmarks)
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33, 160, 158, 133, 153, 144]

        # Извлечение координат для левого и правого глаза
        left_eye_points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in left_eye_indices]
        right_eye_points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in right_eye_indices]

        # Выделение и отображение левого глаза
        x_left, y_left, w_left, h_left = cv2.boundingRect(np.array(left_eye_points))
        left_eye_image = frame[y_left:y_left + h_left, x_left:x_left + w_left]

        # Выделение и отображение правого глаза
        x_right, y_right, w_right, h_right = cv2.boundingRect(np.array(right_eye_points))
        right_eye_image = frame[y_right:y_right + h_right, x_right:x_right + w_right]

        # Визуализация результатов
        cv2.imshow("Left Eye", left_eye_image)
        cv2.imshow("Right Eye", right_eye_image)

    # Прерывание при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()
