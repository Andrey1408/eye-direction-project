import cv2
import mediapipe as mp

# Инициализация объектов для работы с MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Инициализация объекта для работы с веб-камерой
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh() as face_mesh:
    while cap.isOpened():
        # Чтение кадра с веб-камеры
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразование кадра в RGB, так как MediaPipe ожидает входные данные в формате RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Обнаружение ключевых точек лица
        results = face_mesh.process(rgb_frame)

        # Отрисовка сетки лица на кадре
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Изменение размера и цвета точек
                mp_drawing.draw_landmarks(frame, landmarks,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2))

        # Отображение результата
        cv2.imshow('Face Mesh', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()