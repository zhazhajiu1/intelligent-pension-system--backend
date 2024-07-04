import cv2
import dlib
import numpy as np
import pickle
from sklearn import neighbors

PREDICTOR_PATH = 'data_dlib/shape_predictor_68_face_landmarks.dat'
SKIP_FRAMES = 1
# 距离阈值threshold区分已知和未知的特征向量,可调
def recognize_faces_from_camera(model_path='models/knn_model.clf', predictor_path= PREDICTOR_PATH, threshold=0.4, skip_frames=SKIP_FRAMES):
    # 加载训练好的 KNN 模型
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    # 加载训练好的 SVM 模型
    with open(model_path, 'rb') as f:
        svm_clf = pickle.load(f)

    # 初始化 Dlib 的人脸检测器和特征预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1('data_dlib/dlib_face_recognition_resnet_model_v1.dat')

    # 从摄像头开始视频捕获
    cap = cv2.VideoCapture(0)
    frame_count = 0  # 帧计数器

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每隔 skip_frames 帧处理一次
        if frame_count % skip_frames == 0:
            # 将帧转换为灰度图以进行人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                shape = predictor(gray, face)
                encoding = np.array(face_rec_model.compute_face_descriptor(frame, shape))

                # KNN
                closest_distances = knn_clf.kneighbors([encoding], n_neighbors=1)
                is_recognized = closest_distances[0][0][0] <= threshold
                # SVM
                # probabilities = svm_clf.predict_proba([encoding])
                # max_prob = np.max(probabilities)
                # is_recognized = max_prob >= threshold

                x, y, w, h = face.left(), face.top(), face.width(), face.height()

                if is_recognized:
                    label = knn_clf.predict([encoding])[0]
                    identity, person_name = label.split('_')
                    color = (0, 255, 0)  # 绿色框表示识别成功
                    # text = label
                    text = f"{identity}: {person_name}"
                else:
                    color = (0, 0, 255)  # 红色框表示识别失败
                    text = "Unknown"

                # 在人脸周围绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # 在人脸下方显示标签
                cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示结果帧
        cv2.imshow('Face Recognition', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # 增加帧计数器
    # 释放摄像头并关闭所有 OpenCV 窗口
    cap.release()
    cv2.destroyAllWindows()

# 从摄像头进行人脸识别
recognize_faces_from_camera(model_path='models/knn_model.clf', predictor_path= PREDICTOR_PATH, skip_frames=SKIP_FRAMES)
# recognize_faces_from_camera(model_path='models/svm_model.clf', predictor_path= PREDICTOR_PATH, skip_frames=SKIP_FRAMES)