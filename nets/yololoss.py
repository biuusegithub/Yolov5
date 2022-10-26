import math
from copy import deepcopy
from functools import partial
from unittest import result

import numpy as np
import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask=[[6,7,8], [3,4,5], [0,1,2]], label_smoothing=0):
        super().__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing
        self.threshold = 4

        # 权值
        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1 * (input_shape[0] * input_shape[1]) / (640 ** 2)
        self.cls_ratio = 0.5 * (num_classes / 80)

        self.cuda = cuda


    def get_pred_boxes(self, layer, x, y, h, w, targets, scaled_anchors, in_h, in_w):
        # targets: 真实框的标签情况 （batch_size, num_ground_true, 5）
        batch_size = len(targets)

        # 生成网格
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(batch_size * len(self.anchors_mask[layer])), 1, 1).view(x.shape).type_as(x)

        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(batch_size * len(self.anchors_mask[layer])), 1, 1).view(y.shape).type_as(x)

        # 生成先验框高宽
        scaled_anchors_layer = np.array(scaled_anchors)[self.anchors_mask[layer]]   
        anchor_w = torch.Tensor(scaled_anchors_layer).index_select(1, torch.LongTensor([0])).type_as(x)
        anchor_h = torch.Tensor(scaled_anchors_layer).index_select(1, torch.LongTensor([1])).type_as(x)

        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后先验框中心和高宽
        pred_boxes_x = torch.unsqueeze(x * 2. - 0.5 + grid_x, -1)
        pred_boxes_y = torch.unsqueeze(y * 2. - 0.5 + grid_y, -1)
        pred_boxes_w = torch.unsqueeze((w * 2) ** 2 * anchor_w, -1)
        pred_boxes_h = torch.unsqueeze((h * 2) ** 2 * anchor_h, -1)
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)

        return pred_boxes

    
    # 使数据在 min 到 max 之间, 小于 min 的变为 min, 大于 max 的变为 max
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (t <= t_max).float() * t + (t > t_max).float() * t_max

        return result    

    
    # 平方损失
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)


    # 二元交叉熵损失, 公式: -target*log(pred) - (1-target)*log(1-pred)
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1-epsilon)
        out = -target * torch.log(pred) - (1.0-target) * torch.log(1.0-pred)

        return out


    def box_giou(self, box1, box2):
        # box: (batch, feat_w, feat_h, anchor_num, (x、y、w、h))
        # [注意] 数字图像坐标系是 y轴箭头向下, x轴箭头向右

        # 预测框左上角右下角坐标
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxes = box1_xy + box1_wh / 2.

        # 真实框左上角右下角坐标
        box2_xy = box1[..., :2]
        box2_wh = box1[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxes = box2_xy + box2_wh / 2.

        # 求 IOU
        intersect_mins = torch.max(box1_mins, box2_mins)
        intersect_maxes = torch.min(box1_maxes, box2_maxes)
        # 若存在负值取 0
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box1_wh[..., 0] * box1_wh[..., 1] + box2_wh[..., 0] * box2_wh[..., 1] - intersect_area
        iou = intersect_area / union_area

        # 求出 c 的面积 -> 即是包裹两框的最小框(根据公式)
        c_mins = torch.min(box1_mins, box2_mins)
        c_maxes = torch.max(box1_maxes, box2_maxes)
        c_wh = torch.max(c_maxes - c_mins, torch.zeros_like(c_maxes))
        c_area = c_wh[..., 0] * c_wh[..., 1]

        giou = iou - (c_area - union_area) / c_area

        return giou

    
    # 标签平滑：soft标签 -> 缓解过拟合
    # 对于标签, 常用one-hot做hard标签, 然后用预测概率去拟合one-hot的真实概率, 这样会导致类别之间的差异会加大
    # 从而导致模型过于相信预测的类别, 因为只有0 or 1的类别, 要么全信要么不信, 使模型过于自信导致过拟合
    # 相当于分了点概率给其他类, 让标签没这么绝对化, 给学习留些泛化空间
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return (1.0 - label_smoothing) * y_true + label_smoothing / num_classes


    def forward(self, layers, images, targets=None, y_true=None):
        # layers: 第几个有效特征层
        # images:（ batch_size, 3*(5+num_classes), 20, 20）、（ batch_size, 3*(5+num_classes), 40, 40）、（ batch_size, 3*(5+num_classes), 80, 80）
        # targets: 真实框的标签情况 （batch_size, num_ground_true, 5）

        batch_size = images.size(0)
        in_h = images.size(2)
        in_w = images.size(3)

        # 计算步长, 一个特征点对应原图多少个像素, 即特征点的信息浓缩
        stride_h, stride_w = self.input_shape[0] / in_h, self.input_shape[0] / in_w

        # 相应缩小anchors大小
        scaled_anchors = [(a_w/stride_w, a_h/stride_h) for a_w, a_h in self.anchors]

        # 调整输入 images 的维度顺序
        # batch_size, 3, h, w, 5+num_classes
        prediction = images.view(batch_size, len(self.anchors_mask[layers]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 对先验框参数进行缩放
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])

        # 先验框高宽进行调整
        h = torch.sigmoid(prediction[..., 2]) 
        w = torch.sigmoid(prediction[..., 3]) 
            
        # 置信度, 判断是否有物体落入框内
        conf = torch.sigmoid(prediction[..., 4])
            
        # 80个类别各自的置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 对预测结果进行解码
        pred_boxes = self.get_pred_boxes(layers, x, y, h, w, targets, scaled_anchors, in_h, in_w)

        if self.cuda:
            y_true = y_true.type_as(images)


        loss = 0
        # y_true: (3, h, w, 5+num_classes)
        # 统计下有多少个框是有物体的
        n = torch.sum(y_true[..., 4] == 1)
        if n != 0:
            # 计算预测结果和真实结果的giou
            giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(images)

            # 计算对应有真实框的先验框的giou损失, 即正例损失
            loss_loc = torch.mean((1 - giou)[y_true[..., 4] == 1])

            # loss_cls计算对应有真实框的先验框的分类损失
            loss_cls = torch.mean(self.BCELoss(pred_cls[y_true[..., 4] == 1], self.smooth_labels(y_true[..., 5:][y_true[..., 4] == 1], self.label_smoothing, self.num_classes)))
            
            # 最后加权相加
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

            # 计算置信度loss
            # torch.where(condition, a, b), 若满足condition, 输出a, 反之输出b
            tobj = torch.where(y_true[..., 4] == 1, giou.detach().clamp(0), torch.zeros_like(y_true[..., 4]))

        else:
            tobj = torch.zeros_like(y_true[..., 4])

        loss_conf = torch.mean(self.BCELoss(conf, tobj))

        # 最后再加权一下
        loss += loss_conf * self.balance[layers] * self.obj_ratio

        return loss


# 官方函数
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model
    
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

# 权重平滑
class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



