import os
import time
import tqdm
import random
import logging
import datetime
import numpy as np
import albumentations as album

import  torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

from utils import *
from src.datasets.dataset import *
from src.models.ResFormer_MTL_Parallel import *
from src.losses import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True # type: ignore
torch.backends.cudnn.deterministic = True # type: ignore

### Training settings
mode = 'train'
train_batch_size = 8
seg_num_classes = 4
cls_num_classes = 3
base_lr = 0.0001
image_size = 512
dataset = 'eyes'
machine = 'S'
base_dir = './data/{}/{}/'.format(dataset, machine)
train_num = 8
max_epoch = 150

### Create model
model = res34_swin_MS(image_size, seg_num_classes, cls_num_classes)
model_name = 'P_ResFormer_woFEM'
test_model_name = 'ResFormer'

### Log file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s,%(lineno)d: %(message)s\n',
                    datefmt='%Y-%m-%d(%a)%H:%M:%S',
                    filename=os.path.join(f'./ckpt_mtl/{dataset}/{seed}/{machine}/log_{model_name}.txt'),
                    filemode='a')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
information = 'Batch size: {}'.format(train_batch_size) + ' LR {}'.format(base_lr) + \
            ' Img size {}'.format(image_size)

logging.info(information)

model = DataParallel(model)
model.cuda()

# Per-vendor class-balance weights for the classification loss (C / S / T)
weight_bce = torch.FloatTensor([0.43, 0.30, 0.27]).cuda() # S
ce_loss_func = nn.MultiLabelSoftMarginLoss(weight=weight_bce)
dice_loss_func = Dice_and_FocalLoss(n_classes=4)

optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999),
                        eps=1e-08, weight_decay=3e-5, amsgrad=False)

train_transform = album.Compose([album.HorizontalFlip(p=0.5),
                                album.Rotate(limit=25),
                                album.Resize(height=image_size, width=image_size, always_apply=True),
                                ])

train_data = OCT_Dataset(base_dir + 'train/', 'train.txt', train_num, image_size, dataset, train_transform)
trainloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)

print('train len:', len(trainloader))

### Training
iter_num = 0
max_iterations =  max_epoch*len(trainloader)
best_IoU = 0.0
best_acc = 0.0
epoch_num = 0
counter = 0
alpha = 0.25

while epoch_num < max_epoch:
    model.train()
    train_acc = 0
    loss_seg = 0
    loss_cls = 0
    loss_seg_val = 0
    loss_cls_val = 0
    test_acc = 0
    start_time = time.time()

    for images, labels_s, labels_c, name in tqdm.tqdm(trainloader):
        images, labels_s, labels_c = images.cuda(), labels_s.cuda(), labels_c.cuda()

        out_s, out_c = model(images)
        out_s = torch.softmax(out_s, dim=1)
        loss_s = dice_loss_func(out_s, labels_s.long())
        loss_c = ce_loss_func(out_c, labels_c)
        loss = loss_s + alpha*loss_c
        loss_seg += loss_s.data.cpu().numpy()
        loss_cls += loss_c.data.cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr = base_lr * (1.0 - iter_num / max_iterations)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        iter_num = iter_num + 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    model.eval()

    if epoch_num % 1 == 0:
        information = 'Epoch: {} / {}'.format(epoch_num, max_epoch) + ' train time {}'.format(total_time_str) + \
                    ' Loss seg {:4f}'.format(loss_seg/len(trainloader)) + ' Loss cls {:4f}'.format(loss_cls/len(trainloader)) + \
                        '  LR {:4f}'.format(lr)
        logging.info(information)

        save_dir='./result/'
        val_transform = album.Compose([album.Resize(height=image_size, width=image_size, always_apply=True)])
        db_val = OCT_Dataset(base_dir + 'val/', 'val.txt', train_num, image_size, dataset, transform=val_transform)
        valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

        preds_c = []
        targets_c = []

        evaluator_IRF = Evaluator()
        evaluator_SRF = Evaluator()
        evaluator_PED = Evaluator()

        with torch.no_grad():
            for images, labels_s, labels_c, name in tqdm.tqdm(valloader):
                images, labels_s, labels_c = images.cuda(), labels_s.cuda(), labels_c.cuda()

                _pred_s, _pred_c = model(images)
                loss_s_val = dice_loss_func(torch.softmax(_pred_s, dim=1), labels_s.long())
                loss_c_val = ce_loss_func(_pred_c, labels_c)
                loss_seg_val += loss_s_val.data.cpu().numpy()
                loss_cls_val += loss_c_val.data.cpu().numpy()
                pred_s = torch.argmax(torch.softmax(_pred_s, dim=1), dim=1)
                pred_s = F.one_hot(pred_s.long(), num_classes=seg_num_classes)
                label_s =  F.one_hot(labels_s.long(), num_classes=seg_num_classes)

                evaluator_IRF.update(pred_s[0,:,:,1], label_s[0,:,:,1].float())
                evaluator_SRF.update(pred_s[0,:,:,2], label_s[0,:,:,2].float())
                evaluator_PED.update(pred_s[0,:,:,3], label_s[0,:,:,3].float())

                preds_c.append(torch.sigmoid(_pred_c))
                targets_c.append(labels_c)

        Recall_irf, Pre_irf, Acc_irf, Dice_irf, IoU_irf = evaluator_IRF.show(False)
        Recall_srf, Pre_srf, Acc_srf, Dice_srf, IoU_srf = evaluator_SRF.show(False)
        Recall_ped, Pre_ped, Acc_ped, Dice_ped, IoU_ped = evaluator_PED.show(False)

        Dice =  (Dice_irf + Dice_srf + Dice_ped)/3
        IoU =  (IoU_irf + IoU_srf + IoU_ped)/3

        preds_c = torch.cat(preds_c).cpu().numpy()
        targets_c = torch.cat(targets_c).cpu().numpy()
        final_preds = preds_c > 0.5
        acc, sen, spe = evaluation_multilabel(targets_c, final_preds, num_classes=cls_num_classes, is_train=True)

        if IoU > best_IoU:
            best_IoU = IoU
            counter = 0
            torch.save(model.state_dict(), f'./ckpt_mtl/{dataset}/{seed}/{machine}/' + model_name + '_best_iou.pth')

        epoch_num += 1
        result_info = " Acc: " + "%.2f" % acc + " Sen: " + "%.2f" % sen + \
                " Spe: " + "%.2f" % spe + " Dice: " + "%.2f" % Dice + " IoU: " + "%.2f" % IoU

        logging.info(result_info)
