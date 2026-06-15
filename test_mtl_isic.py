import time
import tqdm
import albumentations as album
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

from torch.nn import DataParallel

from utils import *
from src.models.ResFormer_MTL_Parallel import *
from src.datasets.dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
seed = 2023
seg_num_classes = 1
cls_num_classes = 3
batch_size = 1
image_size = 224
machine = 'S'
threshold = 0.5
train_num = 1

dataset = 'isic'
save_dir = f'./result/{dataset}/ours_p/'
base_dir = f'./data/{dataset}/'

val_transform = album.Compose([album.Resize(height=image_size, width=image_size, always_apply=True)])
db_val = ISIC_Dataset(base_dir + 'test/', 'test.txt', train_num, image_size, dataset, transform=None)
valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=8)

model_name = 'ResFormer_isic_best'
model = res34_swin_MS(image_size, seg_num_classes, cls_num_classes)
model = DataParallel(model)

print('Test ', './ckpt_mtl/{}/{}/'.format(dataset, seed, machine) + model_name + '.pth')
model.load_state_dict(torch.load(f'./ckpt_mtl/{dataset}/{seed}/{machine}/{model_name}.pth'))
model.cuda()
model.eval()

evaluator_MA = Evaluator()

start_time = time.time()
pred_c = []
target_c = []
with torch.no_grad():
    for images, labels_s, labels_c, name in tqdm.tqdm(valloader):
        images, labels_s, labels_c = images.cuda(), labels_s.cuda(), labels_c.cuda()

        _pred_s, _pred_c = model(images)

        pred_s = torch.sigmoid(_pred_s) > 0.5
        pred_s = pred_s.permute(0, 2, 3, 1)
        labels_s_1hot = F.one_hot(labels_s.long(), num_classes=2)

        _pred_c = torch.softmax(_pred_c, dim=1)
        pred_c.append(_pred_c)
        target_c.append(labels_c)

        evaluator_MA.update(pred_s[0,:,:,0], labels_s_1hot[0,:,:,1].float())

        for i in range(batch_size):
            save_results(pred_s.squeeze(-1), save_dir, image_size, image_size, name[i])

    preds_c = torch.cat(pred_c).cpu().numpy()
    targets_c = torch.cat(target_c).cpu().numpy()
    final_preds = preds_c > threshold
    a_l, se_l, sp_l, acc, sen, spe = evaluation_multilabel(\
        targets_c, final_preds, num_classes=cls_num_classes, is_train=False)
    auc_scores = roc_auc_score(targets_c, preds_c)
    Recall_ma, Pre_ma, Acc_ma, Dice_ma, IoU_ma = evaluator_MA.show(False)

    Dice = Dice_ma
    IoU = IoU_ma

print('Acc: {:.2f}, Sen: {:.2f}, Spe: {:.2f}, AUC: {:.2f}'.format(acc, sen, spe, auc_scores*100))
print('Dice: {:.2f}, IoU: {:.2f}'.format(Dice, IoU))
