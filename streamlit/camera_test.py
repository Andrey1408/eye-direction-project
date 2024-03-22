# import cv2
# import mediapipe as mp

# # Инициализация объектов для работы с MediaPipe
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils
# denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# # Инициализация объекта для работы с веб-камерой
# cap = cv2.VideoCapture(0)

# with mp_face_mesh.FaceMesh() as face_mesh:
#     while cap.isOpened():
#         # Чтение кадра с веб-камеры
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Преобразование кадра в RGB, так как MediaPipe ожидает входные данные в формате RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Обнаружение ключевых точек лица
#         results = face_mesh.process(rgb_frame)

#         # Отрисовка сетки лица на кадре
#         if results.multi_face_landmarks:
#             for landmarks in results.multi_face_landmarks:
#                 # Изменение размера и цвета точек
#                 mp_drawing.draw_landmarks(frame, landmarks,
#                                           landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2))

#         # Отображение результата
#         cv2.imshow('Face Mesh', frame)

#         # Выход по нажатию клавиши 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Освобождение ресурсов
# cap.release()
# cv2.destroyAllWindows()

import cv2

# Загрузка предварительно обученного детектора глаз из OpenCV
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Загрузка предварительно обученного детектора лиц из OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Захват видеопотока с веб-камеры (номер 0 обычно соответствует встроенной камере)
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10))

    for (x, y, w, h) in faces:
        # Рисование прямоугольника вокруг лица
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Обнаружение глаз внутри области лица
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Рисование прямоугольника вокруг глаз
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('Driver Attention Monitoring', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
