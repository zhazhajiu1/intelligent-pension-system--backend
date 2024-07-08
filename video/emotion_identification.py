import atexit
import signal
import sys
import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionRecognition:
    def __init__(self, model_path, predictor_path, face_rec_model_path):
        # 加载表情识别模型
        self.model = load_model(model_path)
        # 初始化 Dlib 的人脸检测器和特征预测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

        self.emotions = ['happy', 'normal', 'surprise']
        # self.emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness']

    def recognize_emotion(self, frame, face):
        shape = self.predictor(frame, face)
        aligned_face = dlib.get_face_chip(frame, shape, size=48)
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        aligned_face = aligned_face.reshape(1, 48, 48, 1).astype('float32') / 255.0

        emotion_prediction = self.model.predict(aligned_face)
        max_index = np.argmax(emotion_prediction)
        emotion = self.emotions[max_index]

        return emotion

def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and closed all windows.")

def main():
    model_path = 'emotion_recognition_model.h5'
    predictor_path = '../face_recognition/data_dlib/shape_predictor_68_face_landmarks.dat'
    face_rec_model_path = '../face_recognition/data_dlib/dlib_face_recognition_resnet_model_v1.dat'

    emotion_recognition = EmotionRecognition(model_path, predictor_path, face_rec_model_path)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # 寄存器清理
    atexit.register(release_resources, cap)
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))  # Catch Ctrl+C
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))  # Catch termination signal

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            frame = emotion_recognition.recognize_emotion(frame)

            cv2.imshow('Emotion Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Process interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Released camera and closed all windows.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
