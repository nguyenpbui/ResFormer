import os
import cv2
import copy
import torch
import random
import itertools
import numpy as np
from scipy import ndimage

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import Sampler

from sklearn.metrics import multilabel_confusion_matrix

class Evaluator:
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.MAE = list()
        self.Recall = list()
        self.Precision = list()
        self.Accuracy = list()
        self.Dice = list()
        self.IoU = list()

    def evaluate(self, pred, gt):
        eps = 1e-5
        pred_bin = (pred >= 0.5).float().cuda()
        pred_bin_inv = (pred_bin == 0).float().cuda()
        gt_bin = (gt >= 0.5).float().cuda()
        gt_bin_inv = (gt_bin == 0).float().cuda()

        if_present = torch.sum(gt_bin)

        TP = pred_bin.mul(gt_bin).sum().cuda(0)
        FP = pred_bin.mul(gt_bin_inv).sum().cuda(0)
        TN = pred_bin_inv.mul(gt_bin_inv).sum().cuda(0)
        FN = pred_bin_inv.mul(gt_bin).sum().cuda(0)

        Recall = TP / (TP + FN)
        Precision = TP / (TP + FP)
        Accuracy = (TP + TN) / (TP + FP + FN + TN)
        Dice = (2*TP)/(2*TP + FP + FN)
        IoU = (TP) / (TP + FP + FN)

        if if_present:
            return  Recall.data.cpu().numpy().squeeze(), \
                    Precision.data.cpu().numpy().squeeze(), \
                    Accuracy.data.cpu().numpy().squeeze(), \
                    Dice.data.cpu().numpy().squeeze(), \
                    IoU.data.cpu().numpy().squeeze()
        else:
            return None, None, None, None, None

    def update(self, pred, gt):
        recall, precision, accuracy, dice, ioU = self.evaluate(pred, gt)
        self.Recall.append(recall)
        self.Precision.append(precision)
        self.Accuracy.append(accuracy)
        self.Dice.append(dice)
        self.IoU.append(ioU)

    def show(self, flag = True):
        if flag == True:
            print("Recall:", "%.2f" % (np.mean(self.Recall)*100), "\
                    Pre:", "%.2f" % (np.mean(self.Precision)*100),\
                    "  Acc:", "%.2f" % (np.mean(self.Accuracy)*100),"\
                        Dice:", "%.2f" % (np.mean(self.Dice)*100),"\
                        IoU:" , "%.2f" % (np.mean(self.IoU)*100))
            print('\n')

        return  self.calculate_mean(self.Recall)*100,\
                self.calculate_mean(self.Precision)*100,\
                self.calculate_mean(self.Accuracy)*100,\
                self.calculate_mean(self.Dice)*100,\
                self.calculate_mean(self.IoU)*100

    def calculate_mean(self, input):
        if None in input:
            total = sum(filter(None, input))
            cnt = len(input) - input.count(None)
            return total/cnt
        else:
            return np.mean(input)

def evaluation_multilabel(y_true, y_pred, num_classes=5, is_train=True):
    N, C = y_true.shape
    pred_extended = np.c_[y_pred, np.zeros(N)]
    true_extended = np.c_[y_true, np.zeros(N)]

    pred_extended[:,-1] = 1 - np.max(y_pred, axis=1)
    true_extended[:,-1] = 1 - np.max(y_true, axis=1)

    output = multilabel_confusion_matrix(true_extended, pred_extended)

    acc = []
    sen = []
    spe = []
    tps, tns, fps, fns = 0, 0, 0, 0
    for i in range(num_classes):
        tp, tn, fp, fn = output[i,1,1], output[i,0,0], output[i,0,1], output[i,1,0]
        acc.append((tp+tn)/(tp+tn+fp+fn)*100)
        sen.append(tp/(tp+fn)*100)
        spe.append(tn/(tn+fp)*100)
        tps += tp
        tns += tn
        fps += fp
        fns += fn

    if not is_train:
        return acc, sen, spe, (tps+tns)/(tps+tns+fps+fns)*100, tps/(tps+fns)*100, tns/(tns+fps)*100
    else:
        return (tps+tns)/(tps+tns+fps+fns)*100, tps/(tps+fns)*100, tns/(tns+fps)*100

def one_hot_encoder(input_tensor, n_classes=2):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        temp_prob = torch.unsqueeze(temp_prob, 1)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def save_results(pred, save_dir, h, w, j):
    predictions = pred.cpu().numpy()
    test_num= len(predictions)
    for i in range(test_num):
        pred = predictions[i]
        pred_vis = np.zeros((h, w, 3), np.uint8)
        pred_vis[pred==1] = [255,0,0]
        pred_vis[pred==2] = [0,255,0]
        pred_vis[pred==3] = [0,0,255]
        cv2.imwrite(save_dir + str(j) + '.png', pred_vis[:,:,::-1])
