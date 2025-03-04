import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yololoss import (ModelEMA, YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init)

from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes, show_config
from utils.utils_fit import fit_one_epoch



def Train(classes_path, anchors_path, model_path, train_annotation_path, val_annotation_path, phi, UnFreeze_Epoch, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], 
            Cuda=True, distributed=False, fp16=True, input_shape=[640, 640], mosaic=True, mixup=True, special_aug_ratio=0.7, 
            label_smoothing=0, Init_Epoch=0, Freeze_Epoch=50, Freeze_Train=True, Init_lr=1e-2, 
            optimizer_type="sgd", momentum=0.937, weight_decay=5e-4, lr_decay_type="cos", save_period=10, save_dir="logs", 
            eval_flag=True, eval_period=10, num_workers=4):

    mosaic_prob = 0.5
    mixup_prob = 0.5
    Freeze_batch_size = 16
    Unfreeze_batch_size = 8
    Min_lr = Init_lr * 0.01
    num_gpus = torch.cuda.device_count()

    # 是否分布式
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", num_gpus)
    
    # local_rank：DistributedDataParallel 单机多卡训练, 一般不改动
    # 由于一般是单卡, 所以不用分布式运算, 单卡local_rank为0       
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0


    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)

    # 定义模型
    model = YoloBody(anchors_mask, num_classes, phi, input_shape=input_shape)

    # 权重初始化
    weights_init(model)

    # 加载模型
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    # 定义loss
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)

    # 记录loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
    
    # 是否使用混合精度
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            
    # 权值平滑
    ema = ModelEMA(model_train)

    # 读取数据集对应的txt
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        # 打印下超参数
        show_config(
            classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, 
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, 
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, 
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

        wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
        total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch

        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小, 无法进行训练, 请扩充数据集。')

            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时, 建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d, Unfreeze_batch_size为%d, 共训练%d个Epoch, 计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d, 小于建议总步长%d, 建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 判断当前batch_size, 自适应调整学习率
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

        optimizer = {
            'adam' : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd' : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
        }[optimizer_type]

        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})


        # lr下降的公式
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        

        # 判断每一个epoch的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * Init_Epoch
        

        train_dataset = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, 
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
        val_dataset = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, 
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size = batch_size // num_gpus
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        train_dataloader = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        test_dataloader = DataLoader(val_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)


        # 记录eval的map曲线
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, Cuda, 
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        

        # 开始模型训练, 训练Init_Epoch ~ UnFreeze_Epoch代
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                    
                if ema:
                    ema.updates = epoch_step * epoch

                if distributed:
                    batch_size = batch_size // num_gpus
                    

                UnFreeze_flag = True

            train_dataloader.dataset.epoch_now = epoch
            test_dataloader.dataset.epoch_now = epoch

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader, test_dataloader, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()






if __name__ == '__main__':

    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    model_path = 'model_data/yolov5_s.pth'
    phi = 's'
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'
    UnFreeze_Epoch = 23      # 训练的总epoch数

    Train(classes_path, anchors_path, model_path, train_annotation_path, val_annotation_path, phi, UnFreeze_Epoch)





















