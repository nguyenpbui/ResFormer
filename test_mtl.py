import time
import tqdm
import albumentations as album
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

from torch.nn import DataParallel

from utils import *
from src.models.ResFormer_MTL import *
from src.datasets.dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
seed = 2023
seg_num_classes = 4
cls_num_classes = 3
batch_size = 1
image_size = 512
machine = 'S'
threshold = 0.5
train_num = 1

dataset = 'eyes'
save_dir = f'./result/{dataset}/{machine}/ours_p/'
base_dir = f'./data/{dataset}/{machine}/'

val_transform = album.Compose([album.Resize(height=image_size, \
                width=image_size, always_apply=True)])
db_val = OCT_Dataset(base_dir + 'val/', 'val.txt', train_num, image_size, \
                dataset, transform=val_transform)
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)

model_name = 'ResFormer_best'
model = res34_swin_MS(image_size, seg_num_classes, cls_num_classes)
model = DataParallel(model)

print('Test ', f'./ckpt_mtl/{model_name}.pth')
model.load_state_dict(torch.load(f'./ckpt_mtl/{model_name}.pth'))
model.cuda()
model.eval()

evaluator_IRF = Evaluator()
evaluator_SRF = Evaluator()
evaluator_PED = Evaluator()

start_time = time.time()
pred_c = []
target_c = []
with torch.no_grad():
    for images, labels_s, labels_c, name in tqdm.tqdm(valloader):
        images, labels_s, labels_c = images.cuda(), labels_s.cuda(), labels_c.cuda()

        _pred_s, _pred_c = model(images)

        _pred_s = torch.argmax(torch.softmax(_pred_s, dim=1), dim=1)
        pred_s = F.one_hot(_pred_s.long(), num_classes=seg_num_classes)
        labels_s_1hot =  F.one_hot(labels_s.long(), num_classes=seg_num_classes)
        _pred_c = torch.sigmoid(_pred_c)
        pred_c.append(_pred_c)
        target_c.append(labels_c)

        evaluator_IRF.update(pred_s[0,:,:,1], labels_s_1hot[0,:,:,1].float())
        evaluator_SRF.update(pred_s[0,:,:,2], labels_s_1hot[0,:,:,2].float())
        evaluator_PED.update(pred_s[0,:,:,3], labels_s_1hot[0,:,:,3].float())

        save_results(_pred_s, save_dir, image_size, image_size, name[0])

    preds_c = torch.cat(pred_c).cpu().numpy()
    targets_c = torch.cat(target_c).cpu().numpy()
    final_preds = preds_c > threshold
    a_l, se_l, sp_l, acc, sen, spe = evaluation_multilabel(targets_c, final_preds, \
                        num_classes=cls_num_classes, is_train=False) # type: ignore
    auc_scores = roc_auc_score(targets_c, preds_c)

    Recall_he, Pre_he, Acc_he, Dice_he, IoU_he = evaluator_IRF.show(False)
    Recall_se, Pre_se, Acc_se, Dice_se, IoU_se = evaluator_SRF.show(False)
    Recall_ex, Pre_ex, Acc_ex, Dice_ex, IoU_ex = evaluator_PED.show(False)

    Dice =  (Dice_he + Dice_se + Dice_ex)/3
    IoU =  (IoU_he + IoU_se + IoU_ex)/3

print(a_l, se_l, sp_l)
print('Acc: {:.2f}, Sen: {:.2f}, Spe: {:.2f}, AUC: {:.2f}'.\
        format(acc, sen, spe, auc_scores*100))
print(Dice_he, Dice_se, Dice_ex)
print(IoU_he, IoU_se, IoU_ex)
print('Dice: {:.2f}, IoU: {:.2f}'.format(Dice, IoU))
