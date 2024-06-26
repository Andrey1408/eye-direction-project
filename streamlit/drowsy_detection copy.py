import cv2
import time
import numpy as np
import mediapipe as mp
import tensorflow as tf

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from tensorflow.keras.models import load_model

def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist

def get_ear(landmarks, refer_idxs, frame_width, frame_height):

    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y,
                                             frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate Eye aspect ratio"""

    left_ear, left_lm_coordinates = get_ear(
                                      landmarks,
                                      left_eye_idxs,
                                      image_w,
                                      image_h
                                    )
    right_ear, right_lm_coordinates = get_ear(
                                      landmarks,
                                      right_eye_idxs,
                                      image_w,
                                      image_h
                                    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

def get_mediapipe_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5):

    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
 
    return face_mesh

def plot_eye_landmarks(frame, left_lm_coordinates, 
                       right_lm_coordinates, color
                       ):
    
    frame_copy = frame.copy()

    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame_copy, coord, 2, color, -1)
 
    frame_copy = cv2.flip(frame_copy, 1)
    return frame_copy

def plot_text(image, text, origin, 
              color, font=cv2.FONT_HERSHEY_SIMPLEX, 
              fntScale=0.8, thickness=2
              ):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks.
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }
 
        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR
 
        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()
 
        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds time passed with EAR < EAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
        }
 
        self.EAR_txt_pos = (10, 30)
        self.class_names = ['high_center', 'high_left', 'high_right', 'low_center', 'low_left', 'low_right', 'middle_center', 'middle_left', 'middle_right']
        self.model = tf.keras.models.load_model('./model/eye_model')

    def get_gaze_direction(self, left_eye_coords, right_eye_coords, frame_width, frame_height):
        left_eye_center = np.mean(left_eye_coords, axis=0)
        right_eye_center = np.mean(right_eye_coords, axis=0)

        # Нормализуем координаты к центру глаза
        left_eye_center_norm = left_eye_center - np.array([frame_width / 2, frame_height / 2])
        right_eye_center_norm = right_eye_center - np.array([frame_width / 2, frame_height / 2])

        # Определяем направление взгляда
        if abs(left_eye_center_norm[1]) < 10 and abs(right_eye_center_norm[1]) < 10:
            gaze_direction = "Straight"
        elif left_eye_center_norm[1] < right_eye_center_norm[1]:
            gaze_direction = "Top"
        elif left_eye_center_norm[1] > right_eye_center_norm[1]:
            gaze_direction = "Bottom"
        elif left_eye_center_norm[0] < right_eye_center_norm[0]:
            gaze_direction = "Left"
        else:
            gaze_direction = "Right"

        return gaze_direction

    def process_frame(self, landmarks, frame):

        if landmarks:

            # Извлечение координат для левого и правого глаза
            left_eye_points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in self.eye_idxs['left']]
            right_eye_points = [(int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0])) for i in self.eye_idxs['right']]

            # Выделение левого глаза
            x_left, y_left, w_left, h_left = cv2.boundingRect(np.array(left_eye_points))
            left_eye_image = frame[y_left:y_left + h_left, x_left:x_left + w_left]

            # Выделение правого глаза
            x_right, y_right, w_right, h_right = cv2.boundingRect(np.array(right_eye_points))
            right_eye_image = frame[y_right:y_right + h_right, x_right:x_right + w_right]

        predict_classes = []

        for image in [right_eye_image, left_eye_image]:
            img = np.array(image)
            resized = cv2.resize(img, (100,100), interpolation=cv2.INTER_CUBIC)
            img_r = np.zeros((1,resized.shape[0], resized.shape[1], 3))
            for j in range(3):
                img_r[0,:,:,j] = (resized[:,:,j])
            pred = self.model.predict_on_batch(img_r)
            pred = tf.nn.sigmoid(pred).numpy()
            class_predicted = np.argmax(pred[0])
            predict_classes.append(self.class_names[class_predicted])
        
        return predict_classes

    def run(self):
        # Запустите веб-камеру
        cap = cv2.VideoCapture(0)  # 0 указывает на встроенную веб-камеру, но может потребоваться изменение

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Обработайте кадр
            processed_frame, play_alarm = self.process(frame, thresholds)

            # Покажите кадр в новом окне
            cv2.imshow('Drowsiness Detection', processed_frame.astype(np.uint8))

            # Проверьте, была ли активирована тревога, и выполните соответствующие действия

            # Выход из цикла, если нажата клавиша 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освободите ресурсы
        cap.release()
        cv2.destroyAllWindows()

    def process(self, frame: np.array, thresholds: dict):

            # To improve performance,
            # mark the frame as not writeable to pass by reference.
            frame.flags.writeable = False
            frame_h, frame_w, _ = frame.shape
            DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
            ALM_txt_pos = (10, int(frame_h // 2 * 1.85))
    
            results = self.facemesh_model.process(frame)
    
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                EAR, coordinates = calculate_avg_ear(landmarks,
                                                    self.eye_idxs["left"], 
                                                    self.eye_idxs["right"], 
                                                    frame_w, 
                                                    frame_h
                                                    )
                
                direction = self.process_frame(landmarks, frame)

                frame = plot_eye_landmarks(frame, 
                                        coordinates[0], 
                                        coordinates[1],
                                        self.state_tracker["COLOR"]
                                        )
    
                if EAR < thresholds["EAR_THRESH"]:
    
                    # Increase DROWSY_TIME to track the time period with 
                    # EAR less than the threshold
                    # and reset the start_time for the next iteration.
                    end_time = time.perf_counter()
    
                    self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["start_time"]
                    self.state_tracker["start_time"] = end_time
                    self.state_tracker["COLOR"] = self.RED
    
                    if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                        self.state_tracker["play_alarm"] = True
                        plot_text(frame, "WAKE UP! WAKE UP", 
                                ALM_txt_pos, self.state_tracker["COLOR"])
    
                else:
                    self.state_tracker["start_time"] = time.perf_counter()
                    self.state_tracker["DROWSY_TIME"] = 0.0
                    self.state_tracker["COLOR"] = self.GREEN
                    self.state_tracker["play_alarm"] = False
    
                EAR_txt = f"EAR: {round(EAR, 2)}, Direct: {direction}"
                # DIRECTION_txt = f"{gaze_direction}"
                DROWSY_TIME_txt = f"DROWSY: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
                plot_text(frame, EAR_txt, 
                        self.EAR_txt_pos, self.state_tracker["COLOR"])
                plot_text(frame, DROWSY_TIME_txt, 
                        DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])
    
            else:
                self.state_tracker["start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False
    
                # Flip the frame horizontally for a selfie-view display.
                frame = cv2.flip(frame, 1)
    
            return frame, self.state_tracker["play_alarm"]

thresholds = {
    "EAR_THRESH": 0.18,
    "WAIT_TIME": 2,
}

if __name__ == "__main__":
    video_handler = VideoFrameHandler()
    video_handler.run()
