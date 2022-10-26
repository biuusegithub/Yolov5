import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input, resize_image)
from utils.utils_bbox import DecodeBox


'''
[注] YOLOV5利用先验框与真实目标框的宽高比来定义正例, 宽高比小于阈值即为正例, 同时以允许多个先验框匹配一个真实目标框的方式增大正例的比例,
    并设置偏移量, 用临近网格同时预测同一个目标, 进一步增加正例数量, 使得正例比例大大增加.
'''

class YOLO(nn.Module):

    def __init__(self) :
        super().__init__()
        self.model_path = 'model_data/yolov5_s.pth'
        self.classes_path = 'model_data/coco_classes.txt'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.input_shape = [640, 640]
        self.backbone = 'cspdarknet'
        self.phi = 's'
        self.confidence = 0.5
        self.nms_iou = 0.3
        self.letterbox_image = True
        self.cuda = True

        # 获得种类和先验框的数量
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi, input_shape=self.input_shape)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()

        # 这里注意要把权重也放入gpu, 否则会出现input和weights不在同一区域计算的错误
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def forward(self, image, crop, count):

        image_shape = np.array(np.shape(image)[0:2])

        image = cvtColor(image)

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # 加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            # 将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            # result 第6个参数值为 class_pred
            top_label = np.array(results[0][:, 6], dtype = 'int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        # 图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                # fill为画框内填充的颜色, outline为边框颜色
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        print('{} model, and classes loaded.'.format(self.model_path))
        return image


if __name__ == '__main__':
    classes_path = 'model_data/coco_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]

    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    print(colors[0])