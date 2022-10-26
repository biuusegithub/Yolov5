import os

import torch
from tqdm import tqdm

from utils.utils import get_lr



# 训练一个epoch函数
def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, 
                        epoch_step_val, train_dataloader, test_dataloader, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    # local_rank默认为0表示单卡训练
    if local_rank == 0:
        print('Start Train')
        # python内置的一个进度条库
        # mininterval为最小更新时间
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.2, colour='red')

    model_train.train()

    # 训练集
    for iteration, batch in enumerate(train_dataloader):
        if iteration >= epoch_step:
            break
        
        # targets在 YOLODataset中就是boxes (__getitem__的返回值之一)
        # images为原图、targets为真实框、y_trues为先验框
        images, targets, y_trues = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                y_trues = [ann.cuda(local_rank) for ann in y_trues]

        optimizer.zero_grad()

        if not fp16:
            outputs = model_train(images)
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all  += loss_item
            loss_value = loss_value_all

            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value_all  = 0
                for l in range(len(outputs)):
                    loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                    loss_value_all  += loss_item
                loss_value = loss_value_all

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.2, colour='red')

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    # 验证集
    for iteration, batch in enumerate(test_dataloader):
        if iteration >= epoch_step_val:
            break
        images, targets, y_trues = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                y_trues = [ann.cuda(local_rank) for ann in y_trues]

            optimizer.zero_grad()
            outputs = model_train_eval(images)
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets, y_trues[l])
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        # 保存权值
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))