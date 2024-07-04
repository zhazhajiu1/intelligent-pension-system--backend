import io
import time

import cv2
import dlib
import numpy as np
import pickle

import oss2
from sklearn import neighbors
import imutils
from imutils import paths, face_utils

from video.modelsEntity import Reaction

# 填写您的Access Key ID和Access Key Secret
access_key_id = 'LTAI5tR85Q78cGEzRpagmsjh'
access_key_secret = 'JMIJbsWZP6sBPAkxEaQdONt52NSqBx'
# 填写您的Bucket所在地域
endpoint = 'https://oss-cn-beijing.aliyuncs.com'
# 填写Bucket名称
bucket_name = 'old-care-bucket'

# session = requests.Session()
# session.verify = False
# 创建Bucket对象
bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)


class InteractionDetection:
    def __init__(self, knn_model_path, predictor_path, face_rec_model_path, distance_threshold):
        self.knn_clf = self.load_model(knn_model_path)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        self.distance_threshold = distance_threshold
        self.volunteer_label = "volunteer"
        self.elderly_label = "elderly"

    # 加载训练好的模型(KNN)
    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    # 计算焦距
    # def calibrate(self, calibrationImagePath, knownDistance, knownWidth):
    #     # 使用一张已知距离的图片进行焦距标定
    #     image = cv2.imread(calibrationImagePath)
    #     marker = self.find_marker(image)
    #     focalLength = (marker[1][0] * knownDistance) / knownWidth
    #     print(f'a4实际距离: {knownDistance}cm')
    #     print(f'a4实际宽度: {knownWidth}cm')
    #     print(f'a4像素: {marker[1][0]}px')
    #     print(f'焦距: {focalLength}px')
    #     return focalLength

    # # 使用焦距计算距离
    # def calculate_distance(self, knownWidth, perWidth):
    #     # 实际距离 = 实际宽度 * 焦距 / 像素宽度
    #     distance = (knownWidth * KNOWN_FOCAL_LENGTH) / perWidth
    #     return distance

    # def find_marker(self, image):
    #     # 在图像中寻找目标物体的轮廓
    #     # 将图像转换为灰度图，然后进行模糊处理，以去除图像中的高频噪声
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #     # 使用 Canny 算法进行边缘检测
    #     edged = cv2.Canny(gray, 35, 125)
    #     # 寻找边缘图像中的轮廓，保留最大的一个，假设这是我们图像中的纸张
    #     cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #     cnts = imutils.grab_contours(cnts)
    #     c = max(cnts, key=cv2.contourArea)
    #     # 计算纸张区域的边界框，并返回
    #     return cv2.minAreaRect(c)

    # 计算距离
    def calculate_distance(self, shape1, shape2):
        eye_distance_cm = 8

        eye1_1 = (shape1.part(36).x, shape1.part(36).x)
        eye1_2 = (shape1.part(42).x, shape1.part(42).x)

        nose1 = shape1.part(30).x
        nose2 = shape2.part(30).x
        distance_px = nose1 - nose2
        # 计算像素欧几里德距离
        eye_distance_px = np.linalg.norm(np.array(eye1_2) - np.array(eye1_1))
        px_to_meter = eye_distance_cm / eye_distance_px  # 像素转换
        distance = distance_px * px_to_meter

        # 确保返回距离为正数
        return abs(distance)

    # 检测交互
    def detect_interaction(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度处理
        faces = self.detector(gray)  # 检测所有人脸
        recognized_faces = []  # 存储同一帧内识别出的人脸
        for face in faces:
            # 获取人脸的68个关键点并进行对人脸进行对齐
            # print('获取人脸')
            shape = self.predictor(gray, face)
            face_chip = dlib.get_face_chip(frame, shape, size=150)
            # 计算输入人脸的特征向量
            face_encoding = np.array(self.face_rec_model.compute_face_descriptor(face_chip))

            # KNN
            closest_distances = self.knn_clf.kneighbors([face_encoding], n_neighbors=1)
            is_recognized = closest_distances[0][0][0] <= 0.4

            label = "Unknown"
            if is_recognized:
                label = self.knn_clf.predict([face_encoding])[0]

            # print(f'{label}')
            recognized_faces.append((face, shape, label))

        interactions = []

        # 遍历同一帧内的所有人脸，计算它们之间的距离
        for i, (face1, shape1, label1) in enumerate(recognized_faces):
            for j, (face2, shape2, label2) in enumerate(recognized_faces):
                if i >= j:
                    continue

                # 判断身份是否符合
                if self.volunteer_label in label1 and self.elderly_label in label2 or self.volunteer_label in label2 and self.elderly_label in label1:
                    # distance = self.calculate_distance(face1, face2)
                    distance = self.calculate_distance(shape1, shape2)
                    print(f'{distance}cm')
                    # 1m内则存储交互信息
                    if distance <= self.distance_threshold:
                        # print('检测到交互')
                        interactions.append((face1, face2, distance, label1, label2))


        return recognized_faces, interactions
