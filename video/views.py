from __future__ import unicode_literals

import binascii
import json
import os
import time
import io

from django.http import JsonResponse, StreamingHttpResponse

from user.models import Elderly, Volunteer
from . import redis_connect
from .emotion_identification import EmotionRecognition
from .react_identification import InteractionDetection
from .models import *
import oss2
import requests
import dlib
import numpy as np
import pickle
from sklearn import neighbors

from .modelsEntity import Emotion, Fall, Unknow, Intrusion, Reaction

import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from pathlib import Path
import pathlib
from .detect import Detector  # 调用detect文件的Detector类
from video import tracker  # 调用修改的tracker文件

pathlib.PosixPath = pathlib.WindowsPath

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

# 人脸识别
PREDICTOR_PATH = 'video/data_dlib/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = 'video/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
SKIP_FRAMES = 3
model_path = 'video/models/knn_model.clf'
KNN_MODEL_PATH = 'video/models/knn_model.clf'
# model_path='video/models/svm_model.clf'
DISTANCE_THRESHOLD = 100  # 1米内,判断为交互
KNOWN_FOCAL_LENGTH = 630.0  # 焦距px

#情绪识别
model_path2 = 'video/models/emotion_recognition_model.h5'
predictor_path2 = 'video/data_dlib/shape_predictor_68_face_landmarks.dat'
face_rec_model_path2 = 'video/data_dlib/dlib_face_recognition_resnet_model_v1.dat'
emotion_recognition = EmotionRecognition(model_path2, predictor_path2, face_rec_model_path2)

#摔倒检测
weights = 'fallDetect.pt'  # 模型权重路径

img_size = 640  # 图像大小
stride = 32  # 步长
half = False  # 是否使用半精度浮点数减少内存占用，需要GPU支持
# 计时器和状态标记

fall_detected = False  # 摔倒检测标志

#入侵检测
p1x = 0
p1y = 0
p2x = 0
p2y = 0
p3x = 0
p3y = 0
p4x = 0
p4y = 0


def draw_mask(im):  # 定义一个函数，用于将鼠标绘制的区域绘制到显示的图片上
    # print(p1x)
    # print(p1y)
    # print(p2x)
    # print(p2y)
    # print(p3x)
    # print(p3y)
    # print(p4x)
    # print(p4y)
    pts = np.array([[p1x, p1y],
                    [p2x, p2y],
                    [p3x, p3y],
                    [p4x, p4y]], np.int32)
    cv2.polylines(im, [pts], True, (255, 255, 0), 3)
    return im


def scale_bboxes(det, im_shape, img0_shape):
    height, width = im_shape
    img0_height, img0_width, _ = img0_shape

    # Scale the bounding box coordinates
    det[:, 0] *= (img0_width / width)  # scale width
    det[:, 1] *= (img0_height / height)  # scale height
    det[:, 2] *= (img0_width / width)  # scale width
    det[:, 3] *= (img0_height / height)  # scale height

    return det


def getImgUrl(img):
    if bucket:
        url = bucket.sign_url('GET', img, 10 * 60)
        return url
    else:
        return 'not link'


def getRoleByToken(token):
    result = json.loads(redis_connect.get(token))
    return result.get('role')


def getIDByToken(token):
    result = json.loads(redis_connect.get(token))
    return result.get('ID')


def getUserNameByToken(token):
    result = json.loads(redis_connect.get(token))
    return result.get('UserName')


# Create your views here.
def gen_display(url):
    predictor_path = PREDICTOR_PATH
    threshold = 0.4
    skip_frames = SKIP_FRAMES
    # 加载训练好的 KNN 模型
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    # 加载训练好的 SVM 模型
    with open(model_path, 'rb') as f:
        svm_clf = pickle.load(f)
    # 初始化 Dlib 的人脸检测器和特征预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1('video/data_dlib/dlib_face_recognition_resnet_model_v1.dat')
    camera = cv2.VideoCapture(url)
    frame_count = 0  # 帧计数器
    unknow_count = 0
    emotion_count = 0
    # 循环读取视频帧
    while camera.isOpened():
        # 读取一帧图像
        ret, frame = camera.read()
        frame_count += 1
        if frame_count % skip_frames == 0:
            if ret:
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
                        # 表情识别
                        emotion = emotion_recognition.recognize_emotion(frame, face)
                        # 在人脸周围绘制矩形框
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        # 在人脸上方显示表情标签
                        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        # 在人脸下方显示身份标签
                        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        emotion_count += 1
                        if identity == 'elderly' and emotion == 'happy' and(emotion_count == 0 or emotion_count % 10 == 0):
                            screenshot = frame.copy()
                            # 获取当前时间
                            getTime = time.strftime('%H:%M:%S', time.localtime())
                            # print(getTime)
                            # 生成截图文件名
                            filename = 'emotion_{}.png'.format(getTime)

                            # 将图像写入内存缓冲区
                            is_success, buffer = cv2.imencode(".png", screenshot)
                            io_buf = io.BytesIO(buffer)

                            # 填写上传到 OSS 后的文件名
                            oss_file_name = 'old-care/emotion/{}'.format(filename)

                            # 上传文件到 OSS
                            bucket.put_object(oss_file_name, io_buf.getvalue())
                            emotion1 = Emotion(ElderlyName=person_name,
                                               ImgUrl=oss_file_name)
                            emotion1.save()

                    else:
                        color = (0, 0, 255)  # 红色框表示识别失败
                        text = "Unknown"

                        # 在人脸周围绘制矩形框
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        # 在人脸下方显示身份标签
                        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        unknow_count += 1
                        if unknow_count == 0 or unknow_count % 10 == 0:
                            screenshot = frame.copy()
                            # 获取当前时间
                            getTime = time.strftime('%H:%M:%S', time.localtime())
                            # 生成截图文件名
                            filename = 'unknow_{}.png'.format(getTime)
                            # 将图像写入内存缓冲区
                            is_success, buffer = cv2.imencode(".png", screenshot)
                            io_buf = io.BytesIO(buffer)

                            # 填写上传到 OSS 后的文件名
                            oss_file_name = 'old-care/unknow/{}'.format(filename)
                            # 上传文件到 OSS
                            bucket.put_object(oss_file_name, io_buf.getvalue())
                            unknow = Unknow(ImgUrl=oss_file_name)
                            unknow.save()

                # 显示结果帧
                # cv2.imshow('Face Recognition', frame)
                # 将图片进行解码
                success1, frame1 = cv2.imencode('.jpeg', frame)

                if success1:
                    # 转换为byte类型的，存储在迭代器中
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame1.tobytes() + b'\r\n')


def video(request):
    url = 0
    return StreamingHttpResponse(gen_display(url), content_type='multipart/x-mixed-replace; boundary=frame')


def getUrl(request):
    return JsonResponse({'code': 20000,
                         'src': {'video1': 'http://192.168.43.105:8080/video/video', }})


def gen_display2(url):
    # 导入YOLOv5模型
    model = attempt_load(weights)
    camera = cv2.VideoCapture(url)
    fall_timer = 0  # 摔倒计时器
    num = 0
    fall_count = 0

    while camera.isOpened():
        ret, frame = camera.read()
        num += 1
        if num % 5 == 0:
            if ret:
                fps, w, h = 30, frame.shape[1], frame.shape[0]
                # 对图像进行Padded resize
                img = letterbox(frame, img_size, stride=stride, auto=True)[0]

                # 转换图像格式
                img = img.transpose((2, 0, 1))[::-1]  # HWC转为CHW，BGR转为RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).float() / 255.0  # 像素值归一化到[0.0, 1.0]
                img = img[None]  # [h w c] -> [1 h w c]

                # 模型推理
                pred = model(img)[0]  # 获取模型输出
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)  # 进行非最大抑制

                # 绘制边框和标签
                det = pred[0] if len(pred) else []  # 检测结果
                annotator = Annotator(frame.copy(), line_width=3, example=str(model.names))

                fall_detected = False  # 每帧默认未检测到摔倒

                if len(det):
                    det = scale_bboxes(det, (img.shape[2], img.shape[3]), frame.shape)
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # 类别索引
                        label = f'Fall ({conf:.2f})' if model.names[c] == 'fall' else f'nofall ({conf:.2f})'
                        color = (0, 0, 255) if model.names[c] == 'fall' else (0, 255, 0)
                        annotator.box_label(xyxy, label, color=color)  # 绘制边框和标签
                        if model.names[c] == 'fall':  # 检查是否为'fall'状态
                            fall_detected = True
                            break  # 一旦检测到摔倒状态就跳出循环

                if fall_detected:
                    fall_timer += 1  # 增加摔倒计时器
                    if fall_timer >= 0.1 * fps:  # 如果摔倒超过2秒（fps=30，每秒30帧）
                        fall_count += 1
                        cv2.putText(frame, "elderly fall le！！", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        print(11111111111111111)
                        if fall_count == 0 or fall_count % 1 == 0:
                            screenshot = frame.copy()
                            # 获取当前时间
                            getTime = time.strftime('%H:%M:%S', time.localtime())
                            print(getTime)
                            # 生成截图文件名
                            filename = 'fall_{}.png'.format(getTime)
                            # 将图像写入内存缓冲区
                            is_success, buffer = cv2.imencode(".png", screenshot)
                            io_buf = io.BytesIO(buffer)

                            # 填写上传到 OSS 后的文件名
                            oss_file_name = 'old-care/fall/{}'.format(filename)
                            # 上传文件到 OSS
                            bucket.put_object(oss_file_name, io_buf.getvalue())
                            fall = Fall(ImgUrl=oss_file_name)
                            fall.save()

                        # cv2.imwrite('fall_detected.jpg', frame)  # 保存截图
                        fall_timer = 0  # 重置计时器
                else:
                    fall_timer = 0  # 重置摔倒计时器

                # 写入视频帧
                frame = annotator.result()

                success1, frame1 = cv2.imencode('.jpeg', frame)

                if success1:
                    # 转换为byte类型的，存储在迭代器中
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame1.tobytes() + b'\r\n')
            else:
                continue


def video2(request):
    url = 'rtmp://8.146.198.150/live'
    return StreamingHttpResponse(gen_display2(url), content_type='multipart/x-mixed-replace; boundary=frame')


def getUrl2(request):
    return JsonResponse({'code': 20000,
                         'src': {'video1': 'http://192.168.43.105:8080/video/video2', }})


def emotionList(request):
    searchName = request.GET.get('UserName')
    searchDate = request.GET.get('Date')

    if not searchDate and not searchName:
        emotions = Emotion.objects.all()
        total = Emotion.objects.count()
    elif searchDate and searchName:
        emotions = Emotion.objects.filter(Created__startswith=searchDate, ElderlyName=searchName)
        total = Emotion.objects.filter(Created__startswith=searchDate, ElderlyName=searchName).count()
    elif not searchDate and searchName:
        emotions = Emotion.objects.filter(ElderlyName=searchName)
        total = Emotion.objects.filter(ElderlyName=searchName).count()
    else:
        emotions = Emotion.objects.filter(Created__startswith=searchDate)
        total = Emotion.objects.filter(Created__startswith=searchDate).count()

    emotion_list = []
    for emotion in emotions:
        url = getImgUrl(emotion.ImgUrl)
        emotion_list.append(
            {
                'ID': emotion.ID,
                'ElderlyName': emotion.ElderlyName,
                'Url': url,
                'Created': emotion.Created,
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': emotion_list}})


def emotionDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    # print(ID)
    # print(request.GET)
    deleteEmotion = Emotion.objects.get(ID=ID)
    # print(deleteUser)
    deleteEmotion.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def emotionDetailByID(request):
    ID = request.GET.get('ID')
    result = Emotion.objects.filter(ID=ID).first()
    result1 = Elderly.objects.filter(UserName=result.ElderlyName).first()
    url = getImgUrl(result.ImgUrl)
    url1 = getImgUrl(result1.ImgUrl)
    return JsonResponse({
        "code": 20000,
        "data": {
            'ID': result.ID,
            'ElderlyID': result1.ID,
            'ElderlyName': result.ElderlyName,
            'Url': url,
            'Created': result.Created,
            'Sex': result1.Sex,
            'Age': result1.Age,
            'Birthday': result1.Birthday,
            'Phone': result1.Phone,
            'Healthy': result1.Healthy,
            'GuardianName': result1.GuardianName,
            'GuardianPhone': result1.GuardianPhone,
            'ElderlyImgUrl': result1.ImgUrl,
            'ElderlyUrl': url1,
            'IsActive': result1.IsActive,
            'ElderlyCreated': result1.Created,
            'Updated': result1.Updated

        }
    })


def fallList(request):
    searchDate = request.GET.get('Date')
    print(searchDate)

    if not searchDate:
        falls = Fall.objects.all()
        total = Fall.objects.count()
    else:
        falls = Fall.objects.filter(Created__startswith=searchDate)
        total = Fall.objects.filter(Created__startswith=searchDate).count()

    fall_list = []
    for fall in falls:
        url = getImgUrl(fall.ImgUrl)
        fall_list.append(
            {
                'ID': fall.ID,
                'Url': url,
                'Created': fall.Created,
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': fall_list}})


def fallDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    print(request.GET)
    deleteFall = Fall.objects.get(ID=ID)
    # print(deleteUser)
    deleteFall.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def unknowList(request):
    searchDate = request.GET.get('Date')

    if not searchDate:
        unknows = Unknow.objects.all()
        total = Unknow.objects.count()
    else:
        unknows = Unknow.objects.filter(Created__startswith=searchDate)
        total = Unknow.objects.filter(Created__startswith=searchDate).count()

    unknow_list = []
    for unknow in unknows:
        url = getImgUrl(unknow.ImgUrl)
        unknow_list.append(
            {
                'ID': unknow.ID,
                'Url': url,
                'Created': unknow.Created,
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': unknow_list}})


def unknowDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    print(request.GET)
    deleteUnknow = Unknow.objects.get(ID=ID)
    # print(deleteUser)
    deleteUnknow.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def gen_display3(url):
    camera = cv2.VideoCapture(url)
    # width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(width)
    # print(height)
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            success1, frame1 = cv2.imencode('.jpeg', frame)
            if success1:
                # 转换为byte类型的，存储在迭代器中
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame1.tobytes() + b'\r\n')


def video3(request):
    url = 0
    return StreamingHttpResponse(gen_display3(url), content_type='multipart/x-mixed-replace; boundary=frame')


def getUrl3(request):
    return JsonResponse({'code': 20000,
                         'src': {'video1': 'http://192.168.43.105:8080/video/video3'},
                         'data': {
                             'width': 640,
                             'height': 480
                         }})


def getXY(request):
    global p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y
    post_body = request.body
    json_param = json.loads(post_body.decode())

    p1x = json_param.get('p1x')
    p1y = json_param.get('p1y')
    p2x = json_param.get('p2x')
    p2y = json_param.get('p2y')
    p3x = json_param.get('p3x')
    p3y = json_param.get('p3y')
    p4x = json_param.get('p4x')
    p4y = json_param.get('p4y')
    # print(p1x)
    # print(p1y)
    # print(p2x)
    # print(p2y)
    # print(p3x)
    # print(p3y)
    # print(p4x)
    # print(p4y)

    return JsonResponse({'code': 20000, 'message': '画框成功'})


def gen_display4(url):
    camera = cv2.VideoCapture(url, cv2.CAP_DSHOW)
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_cnt = 0  # 记录帧数  用于判断帧数是否大于2，进行目标跟踪
    dict_box = dict()
    dic_id = dict()
    det = Detector()  # 生成预测对象
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            frame_cnt += 1
            listboxs = []  # box框
            im, bboxes = det.detect(frame)  # 获取预测结果
            mask = np.zeros((height, width, 3), np.uint8)  # 掩膜区域数值设置为0
            cv2.rectangle(mask, (p1x, p1y), (p3x, p3y), 255, -1)  # 绘制区域，用作判断入侵
            if len(bboxes) > 0:  # 判断是否有预测值
                listboxs = tracker.update(bboxes, im, frame_cnt, dict_box, dic_id)  # 将预测值送入目标跟踪中
                im = tracker.draw_bboxes(im, listboxs, mask)  # 绘制在原图上
                im = draw_mask(im)
            success1, frame1 = cv2.imencode('.jpeg', im)
            if success1:
                # 转换为byte类型的，存储在迭代器中
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame1.tobytes() + b'\r\n')


def video4(request):
    url = 0
    return StreamingHttpResponse(gen_display4(url), content_type='multipart/x-mixed-replace; boundary=frame')


def getUrl4(request):
    return JsonResponse({'code': 20000,
                         'src': {'video1': 'http://192.168.43.105:8080/video/video4', }})


def intrusionList(request):
    searchDate = request.GET.get('Date')

    if not searchDate:
        intrusions = Intrusion.objects.all()
        total = Intrusion.objects.count()
    else:
        intrusions = Intrusion.objects.filter(Created__startswith=searchDate)
        total = Intrusion.objects.filter(Created__startswith=searchDate).count()

    intrusion_list = []
    for intrusion in intrusions:
        url = getImgUrl(intrusion.ImgUrl)
        intrusion_list.append(
            {
                'ID': intrusion.ID,
                'Url': url,
                'Created': intrusion.Created,
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': intrusion_list}})


def intrusionDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    print(request.GET)
    deleteIntrusion = Intrusion.objects.get(ID=ID)
    # print(deleteUser)
    deleteIntrusion.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def gen_display5(url):
    interaction_detector = InteractionDetection(
        knn_model_path=KNN_MODEL_PATH,
        predictor_path=PREDICTOR_PATH,
        face_rec_model_path=FACE_RECOGNITION_MODEL_PATH,
        distance_threshold=DISTANCE_THRESHOLD
    )
    camera = cv2.VideoCapture(url)
    # width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(width)
    # print(height)
    frame_count = 0  # 帧计数器
    reaction_count = -1
    while camera.isOpened():
        ret, frame = camera.read()
        frame_count += 1
        if frame_count % SKIP_FRAMES == 0:
            if ret:
                recognized_faces, interactions = interaction_detector.detect_interaction(frame)

                # 绘制所有人脸框和标签
                for face, shape, label in recognized_faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 绘制交互信息
                for face1, face2, distance, label1, label2 in interactions:
                    x1, y1, w1, h1 = face1.left(), face1.top(), face1.width(), face1.height()
                    x2, y2, w2, h2 = face2.left(), face2.top(), face2.width(), face2.height()

                    # 在人脸上方显示标签和距离信息
                    cv2.line(frame, (x1 + w1 // 2, y1 + h1 // 2), (x2 + w2 // 2, y2 + h2 // 2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{distance:.2f}cm", ((x1 + x2) // 2, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0), 2)

                    reaction_count += 1
                    if reaction_count == 0 or reaction_count % 10 == 0:
                        screenshot = frame.copy()
                        # 获取当前时间
                        getTime = time.strftime('%H:%M:%S', time.localtime())
                        # 生成截图文件名
                        filename = 'reaction_{}.png'.format(getTime)
                        # 将图像写入内存缓冲区
                        is_success, buffer = cv2.imencode(".png", screenshot)
                        io_buf = io.BytesIO(buffer)
                        # 填写上传到 OSS 后的文件名
                        oss_file_name = 'old-care/reaction/{}'.format(filename)
                        # 上传文件到 OSS
                        bucket.put_object(oss_file_name, io_buf.getvalue())
                        identity1, person_name1 = label1.split('_')
                        identity2, person_name2 = label2.split('_')
                        print(identity1, person_name1)
                        print(identity2, person_name2)
                        if identity1 == 'elderly':
                            reaction = Reaction(ElderlyName=person_name1, VolunteerName=person_name2,
                                                ImgUrl=oss_file_name)
                            reaction.save()
                        else:
                            reaction = Reaction(ElderlyName=person_name2, VolunteerName=person_name2,
                                                ImgUrl=oss_file_name)
                            reaction.save()


                success1, frame1 = cv2.imencode('.jpeg', frame)
                if success1:
                    # 转换为byte类型的，存储在迭代器中
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame1.tobytes() + b'\r\n')


def video5(request):
    url = 0
    return StreamingHttpResponse(gen_display5(url), content_type='multipart/x-mixed-replace; boundary=frame')


def getUrl5(request):
    return JsonResponse({'code': 20000,
                         'src': {'video1': 'http://192.168.43.105:8080/video/video5'}})


def reactionList(request):
    searchElderlyName = request.GET.get('ElderlyName')
    searchVolunteerName = request.GET.get('VolunteerName')
    searchDate = request.GET.get('Date')

    if not searchDate and not searchElderlyName and not searchVolunteerName:
        reactions = Reaction.objects.all()
        total = Reaction.objects.count()
    elif searchDate and searchElderlyName and searchVolunteerName:
        reactions = Reaction.objects.filter(Created__startswith=searchDate, ElderlyName=searchElderlyName,
                                          VolunteerName=searchVolunteerName)
        total = Reaction.objects.filter(Created__startswith=searchDate, ElderlyName=searchElderlyName,
                                       VolunteerName=searchVolunteerName).count()
    elif searchDate and not searchElderlyName and not searchVolunteerName:
        reactions = Reaction.objects.filter(Created__startswith=searchDate)
        total = Reaction.objects.filter(Created__startswith=searchDate).count()
    elif not searchDate and searchElderlyName and not searchVolunteerName:
        reactions = Reaction.objects.filter(ElderlyName=searchElderlyName)
        total = Reaction.objects.filter(ElderlyName=searchElderlyName).count()
    elif not searchDate and not searchElderlyName and searchVolunteerName:
        reactions = Reaction.objects.filter(VolunteerName=searchVolunteerName)
        total = Reaction.objects.filter(VolunteerName=searchVolunteerName).count()
    elif not searchDate and searchElderlyName and searchVolunteerName:
        reactions = Reaction.objects.filter(ElderlyName=searchElderlyName,
                                          VolunteerName=searchVolunteerName)
        total = Reaction.objects.filter(ElderlyName=searchElderlyName,
                                       VolunteerName=searchVolunteerName).count()
    elif searchDate and not not searchElderlyName and searchVolunteerName:
        reactions = Reaction.objects.filter(Created__startswith=searchDate,
                                          VolunteerName=searchVolunteerName)
        total = Reaction.objects.filter(Created__startswith=searchDate,
                                       VolunteerName=searchVolunteerName).count()
    elif searchDate and searchElderlyName and not searchVolunteerName:
        reactions = Reaction.objects.filter(Created__startswith=searchDate, ElderlyName=searchElderlyName,)
        total = Reaction.objects.filter(Created__startswith=searchDate, ElderlyName=searchElderlyName,).count()
    else:
        reactions = Reaction.objects.filter(Created__startswith=searchDate)
        total = Reaction.objects.filter(Created__startswith=searchDate).count()

    reaction_list = []
    for reaction in reactions:
        url = getImgUrl(reaction.ImgUrl)
        reaction_list.append(
            {
                'ID': reaction.ID,
                'ElderlyName': reaction.ElderlyName,
                'VolunteerName': reaction.VolunteerName,
                'Url': url,
                'Created': reaction.Created,
            }
        )

    return JsonResponse({'code': 20000,
                         'message': 'success',
                         'data': {'total': total,
                                  'rows': reaction_list}})


def reactionDelete(request):
    token = request.META.get("HTTP_AUTHORIZATION")  # 获取 Authorization 头部
    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]  # 提取实际的 token 部分
    else:
        return JsonResponse({'code': 20003, 'message': '无效的 token 或缺少 token'})
    # print(token)
    role = getRoleByToken(token)
    if role == 'volunteer':
        return JsonResponse({'code': 20003, 'message': '你没有权限'})

    ID = request.GET.get('ID')
    # print(ID)
    # print(request.GET)
    deleteReaction = Reaction.objects.get(ID=ID)
    # print(deleteUser)
    deleteReaction.delete()
    return JsonResponse({'code': 20000, 'message': '删除成功'})


def reactionDetailByID(request):
    ID = request.GET.get('ID')
    print(ID)
    result = Reaction.objects.filter(ID=ID).first()
    print(result)
    result1 = Elderly.objects.filter(UserName=result.ElderlyName).first()
    result2 = Volunteer.objects.filter(UserName=result.VolunteerName).first()
    url = getImgUrl(result.ImgUrl)
    url1 = getImgUrl(result1.ImgUrl)
    url2 = getImgUrl(result2.ImgUrl)
    return JsonResponse({
        "code": 20000,
        "data": {
            'ID': result.ID,
            'ElderlyID': result1.ID,
            'ElderlyName': result.ElderlyName,
            'VolunteerID': result2.ID,
            'VolunteerName': result.VolunteerName,
            'Url': url,
            'Created': result.Created,
            'ElderlySex': result1.Sex,
            'ElderlyAge': result1.Age,
            'ElderlyBirthday': result1.Birthday,
            'ElderlyPhone': result1.Phone,
            'Healthy': result1.Healthy,
            'GuardianName': result1.GuardianName,
            'GuardianPhone': result1.GuardianPhone,
            'ElderlyImgUrl': result1.ImgUrl,
            'ElderlyUrl': url1,
            'ElderlyIsActive': result1.IsActive,
            'ElderlyCreated': result1.Created,
            'ElderlyUpdated': result1.Updated,
            'VolunteerSex': result2.Sex,
            'VolunteerAge': result2.Age,
            'VolunteerPhone': result2.Phone,
            'VolunteerImgUrl': result2.ImgUrl,
            'VolunteerUrl': url2,
            'VolunteerIsActive': result2.IsActive,
            'VolunteerCreated': result2.Created,
            'VolunteerUpdated': result2.Updated
        }
    })
