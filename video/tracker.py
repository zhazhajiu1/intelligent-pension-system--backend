import io
import time

import cv2
import oss2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from video.modelsEntity import Intrusion
from video.sendMessage import send_sms

#阿里云
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

intrusion_count = 0
cfg = get_config()
cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def xyxy2tlwh(x):
    '''
    Convert bounding box format from xyxy to tlwh (top left x, top left y, width, height).
    '''
    y = [None] * len(x)  # Create a list of None values to store results

    # print("x type & y type", type(x), type(y))
    # print("x:", x)

    for i in range(len(x)):
        y[i] = [None] * 4  # Each entry in y is a list of 4 elements
        y[i][0] = x[i][0]  # top left x
        y[i][1] = x[i][1]  # top left y
        y[i][2] = x[i][2] - x[i][0]  # width
        y[i][3] = x[i][3] - x[i][1]  # height

    return y


def draw_bboxes(im, bboxes, mask):  # 绘制目标跟踪框，中心点，和入侵判断。参数（原图，预测框坐标，鼠标绘制的既定区域）
    count = 0  # 用于记录既定区域内入侵人数
    global intrusion_count
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:  # 遍历所有预测框
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # 转int类型用于绘制框
        list_pts = []  # 用于保存中心点框的坐标（个人认为没有多大必要，且写法繁琐，但无关紧要 遂保存）
        check_point_x = int((x2 + x1) / 2)  # 中心点坐标x
        point_radius = 2  # 中心点半径
        check_point_y = int((y2 + y1) / 2)  # 中心点y
        c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))  # 左上右下点

        # 将原图上的预测框中心点区域求掩膜区域同位置的交集，不为0则判断为入侵（注y在前，理解不了的可自行百度
        overlap = cv2.bitwise_and(mask[check_point_y - 3:check_point_y + 3, check_point_x - 3:check_point_x + 3],
                                  im[check_point_y - 3:check_point_y + 3, check_point_x - 3:check_point_x + 3])

        if np.sum(overlap) != 0 and cls_id == 'person':  # 判断交集不为0，且标签为person 判断为入侵
            count += 1  # 入侵人数+1
            cv2.rectangle(im, c1, c2, (0, 0, 255), 2, cv2.LINE_AA)  # 绘制入侵的框，将框的颜色设置为红色
            allname = cls_id + '-ID--' + str(pos_id)  # 将标签名和ID数字符拼接
            t_size = cv2.getTextSize(allname, 0, fontScale=0.5, thickness=2)[0]  # 获取上述文本的长度 用于绘制预测框上的文本框
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)  # 调整文本框右下角坐标
            cv2.rectangle(im, c1, c2, (0, 0, 255), -1)  # filled        #绘制文本框

        else:  # 绘制非既定区域内的框  （绿色框）
            cv2.rectangle(im, c1, c2, (0, 255, 0), 2, cv2.LINE_AA)
            allname = cls_id + '-ID--' + str(pos_id)
            t_size = cv2.getTextSize(allname, 0, fontScale=0.5, thickness=2)[0]
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(im, c1, c2, (0, 255, 0), -1)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])  # 中心点区域
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)
        cv2.fillPoly(im, [ndarray_pts], color=(0, 0, 255))  # 绘制多边形
        list_pts.clear()
        cv2.putText(im, allname, (c1[0], c1[1] - 2), 0, 0.5,  # 将文本写入文本框
                    [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    if count != 0:  # 判断若掩膜区域有目标则写入警告，并绘制统计的人数
        cv2.putText(im, f'Warning:-{count}-peopels', (10, 50), 0, 2, (0, 0, 255), 3, cv2.LINE_AA)
        # TODO:此处截图
        # cv2.imwrite('img/invasion.jpg', im)
        if intrusion_count == 0 or intrusion_count % 10 == 0:
            screenshot = im.copy()
            # 获取当前时间
            getTime = time.strftime('%H:%M:%S', time.localtime())
            # 生成截图文件名
            filename = 'intrusion_{}.png'.format(getTime)
            # 将图像写入内存缓冲区
            is_success, buffer = cv2.imencode(".png", screenshot)
            io_buf = io.BytesIO(buffer)

            # 填写上传到 OSS 后的文件名
            oss_file_name = 'old-care/intrusion/{}'.format(filename)
            # 上传文件到 OSS
            bucket.put_object(oss_file_name, io_buf.getvalue())
            intrusion = Intrusion(ImgUrl=oss_file_name)
            intrusion.save()
            # send_sms("13855732038")
    intrusion_count +=1

    return im  # 返回绘制后的图


# 加入两个空字典(dict_box，dic_id)，1.用于保存中心点的坐标，2.用于记录预测框消失的帧数，用于删除消失目标的中心点坐标
def update(bboxes, im, frame_cnt, dict_box, dic_id):
    bbox_xywh = []
    confs = []
    bboxes2draw = []
    clss = []

    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf, cls_id in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            clss.append(cls_id)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, clss, im)

        if len(outputs) > 0:
            # print(type(outputs))
            # print(outputs)
            bbox_xyxy = outputs[0:4]  # 提取前四列 （坐标）
            identities = outputs[0:-1]  # 提取最后一列 （ID）
            for i in list(dict_box.keys()):  # 遍历保存坐标的字典的键（即id号）
                if i not in identities:  # 如果id 不在预测结果里面，说明id在当前帧消失
                    dic_id[i] += 1  # 则生成一个键值对，用于记录id消失的帧数
                    if dic_id[i] > 10:  # 如果帧数大于十，则在记录中心点坐标的字典中删除此id和坐标，并删除帧数记录
                        dict_box.pop(i)
                        dic_id.pop(i)

            box_xywh = xyxy2tlwh(bbox_xyxy)

            for j in range(len(box_xywh)):  # 遍历坐标
                x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
                y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标
                id = outputs[j][-1]
                center = [x_center, y_center]

                dict_box.setdefault(id, []).append(center)  # 将中心点坐标和id号用字典绑定
                dic_id[id] = 1  # 定义消失计数初始值
                if len(dict_box[id]) > 50:  # 记录只50个中心点坐标，判断长度是否大于50大于则删除第一个
                    dict_box[id].pop(0)

            if frame_cnt > 2:  # 第一帧无法连线，所以设置从第二帧开始，frame_cnt为当前帧号      绘制轨迹
                for key, value in dict_box.items():
                    for a in range(len(value) - 1):
                        index_start = a
                        index_end = index_start + 1
                        cv2.line(im, tuple(map(int, value[index_start])), tuple(map(int, value[index_end])),
                                 # map(int,"1234")转换为list[1,2,3,4]
                                 (0, 0, 255), thickness=2, lineType=8)

        for x1, y1, x2, y2, track_id, cls in list(outputs):
            # x1, y1, x2, y2, track_id = value
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)

            center = (x1, y1, x2, y2, label, track_id)

            bboxes2draw.append((center))
        pass
    pass

    return bboxes2draw


def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    label = ''
    # min_label = ''
    min_dist = -1.0

    for x1, y1, x2, y2, lbl, conf, cls_id in bboxes_xyxy:
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label
