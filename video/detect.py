import torch
import numpy as np
from utils.general import non_max_suppression, scale_boxes  # 非极大抑制
from utils.augmentations import letterbox  # 图像补边操作
from utils.torch_utils import select_device  # 驱动器
from models.experimental import attempt_load  #


class Detector():  # 封装成类，用于获取预测值
    def __init__(self):  # 初始化
        super(Detector, self).__init__()
        self.imgSize = 640
        self.threshold = 0.3
        self.stride = 1
        self.weights = 'yolov5s.pt'#TODO：如果模型位置更改需要修改此处
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, device=self.device)
        model.to(self.device).eval()
        model.float()
        self.model = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names  # 用于将标签概率转化为标签名

    def preprocess(self, img):
        img0 = img.copy()  # 拷贝原图
        img = letterbox(img, new_shape=self.imgSize)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # cv2读出来的是bgr通道需改为rgb通道,在交换维度得到（通道数，宽，高）
        img = np.ascontiguousarray(img)  # 将数组变为一个连续性的数组
        img = torch.from_numpy(img).to(self.device)  # 将数组转换为torch所接受的格式   放到驱动器
        img = img.float()  # 转换成float类型
        img /= 255.0  # 归一化缩放
        if img.ndimension() == 3:  # 判断通道数是不是3
            img = img.unsqueeze(0)  # 是的话就进行升维  增加一个batchsize的维度   传入图像为4维

        return img0, img

    def detect(self, im):
        img0, img = self.preprocess(im)  # 调用上述函数，获取数据处理后的图片
        pred = self.model(img, augment=False)[0]  # 有了图片之后，丢入模型获取预测值
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)  # 将预测值进行非极大值抑制处理
        pred_boxes = []  # 用于保存所需要的值
        for det in pred:  # 遍历非极大值抑制之后的预测值
            if det is not None and len(det):  # 判断非空
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()  # 调用函数用来将预测值的坐标映射到原图

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]  # 将标签概率通过字典转化为标签名
                    if lbl not in ['person', 'bicycle', 'car', 'motorcycle', 'bus',
                                   'truck']:  # 获取所需的标签，即不在列表内的标签不取（这里直接用的yolov5s的权重，八十分类）
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append((x1, y1, x2, y2, lbl, conf, cls_id))  # 将获取的值按元组形式保存到列表

        return im, pred_boxes  # 返回原图 和预测框