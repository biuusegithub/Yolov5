import numpy as np
import torch
from torchvision.ops import nms


# 解码操作：得到特征后如何去调整先验框
class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape

        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # batch_size, 255, 20, 20
            # batch_size, 255, 40, 40
            # batch_size, 255, 80, 80
            # 其中 255 = 3 * (4 + 1 + 80)
            batch_size = input.size(0)
            input_height = input.size(2)
            input_width = input.size(3)

            # 计算原图是每个特征层的多少倍
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width

            # 对anchor的先验框(大小人为设定的)进行缩放调整
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            # 对输入input进行维度调整 (batch_size, len(anchors_mask), height, weight, bbox_attrs)
            prediction = input.view(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            # [..., 0] == [:,:,:,0] , [..., 0, 0] == [:,:,0,0], [..., 0, 0, 0] == [:,0,0,0]
            # (batch_size, 3, height, weight, 5 + 80)
            # 先验框中心位置进行调整
            x = torch.sigmoid(prediction[..., 0])  
            y = torch.sigmoid(prediction[..., 1])

            # 先验框高宽进行调整
            w = torch.sigmoid(prediction[..., 2]) 
            h = torch.sigmoid(prediction[..., 3]) 
            
            # 置信度, 判断是否有物体落入框内
            conf = torch.sigmoid(prediction[..., 4])
            
            # 80个类别各自的置信度
            pred_cls = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            # 生成网格, 每张图每个特征点每个先验框的x、y坐标
            # (batch_size, 3, h, w)
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            # 生成先验框高宽, 把先验框映射到每个特征点上
            # (batch_size, 3, h, w)
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            # 利用预测结果对先验框进行调整, 先调整中心后调整高宽
            # x : -0.5, 1.5 => 负责一定范围的目标的预测
            # y : -0.5, 1.5 => 负责一定范围的目标的预测
            # w : 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # h : 0 ~ 4 => 先验框的宽高调节范围为0~4倍
            # (batch_size, 3, height, weight, 4)
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h

            # 将输出结果按原图大小进行压缩, 即把调整后的坐标大小除以原图的高宽
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            
            # output: (batch_size, 3*h*w, 4+1+80), 其中3*h*w即使所有像素点的anchor数
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            
            outputs.append(output.data)
        
        return outputs


    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        # 对模型输出的box信息(中心x, 中心y, w, h)进行校正,输出基于原图坐标系的box信息(x_min, y_min, x_max, y_max)

        # [::-1] 列表倒置从后往前取值
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        # 把列表转换为数组
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            # resize_shape 指的是原图长宽等比缩放后取整
            resize_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset = (input_shape - resize_shape) / 2. / input_shape
            scale = input_shape / resize_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        
        return boxes


    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):

        # prediction: (batch_size, num_anchors, 85)
        # 85 -> 前两位表示中心点x、y, 后两位表示宽高, 再后一位表示该点是否有物体, 后80为表示个类别概率
        # 由于像素坐标对比直角坐标是倒置的, 所以（中心点x、y, 宽， 高）转换为（左上角x、y, 右下角x、y）
        box_corner = prediction.new(prediction.shape)

        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]

        for i, image_pred in enumerate(prediction):
            # image_pred: (num_anchors, 85)
            # torch.max() 返回两个值 [value, index]
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

            # image_pred[:, 4]为该点是否有物体, 有为1、无为0, class_conf[:, 0]为该点所有种类中的最大概率
            # squeeze() 把空的剔除
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

            # 根据置信度对预测结果进行筛选
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]

            if not image_pred.size(0):
                continue

            # detection：(num_anchors, 7)
            # 7: (x1, y1, x2, y2, obj, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            # 获取预测结果中包含的所有类
            all_labels = detections[:, -1].cpu().unique()    # .unique(): 从小到大排序

            if prediction.is_cuda:
                all_labels = all_labels.cuda()
                detections = detections.cuda()

            for index in all_labels:
                detections_class = detections[detections[:, -1] == index]

                # 直接使用官方的nms, 也可以自己实现
                # 筛选出一定区域内, 属于同一种类的得分最大的框
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4] * detections_class[:, 5],
                    nms_thres
                )
                max_detections = detections_class[keep]
                
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        
        return output
