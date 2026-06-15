import os
import cv2
import torch
import random
import itertools
import numpy as np
import pandas as pd
from scipy import ndimage

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset, DataLoader

class OCT_Dataset(Dataset):
    def __init__(self, root, list_name, train_num, image_size, dataset, transform=None):
        self.root = root
        self.list_path = self.root + list_name
        self.h, self.w = image_size, image_size
        self.dataset = dataset
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.transform = transform
        self.files = []
        for name in self.img_ids:
            if self.dataset == 'skin':
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            elif self.dataset == 'polyp' or self.dataset == 'eyes':
                img_file = os.path.join(self.root, "images/%s.png" % name)
            else:
                img_file = os.path.join(self.root, "images/%s.jpg" % name)

            label_file = os.path.join(self.root, "masks/%s.png" % name)
            self.files.append({"img": img_file, "label": label_file, "name": name})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) # type: ignore
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE) # type: ignore

        if self.transform is not None:
            sample = self.transform(image=image, mask=label)
            image, label = sample['image'], sample['mask']

        image = cv2.resize(image,(self.w, self.h), interpolation = cv2.INTER_NEAREST)
        if self.dataset == 'idrid' or self.dataset == 'eyes':
            label = cv2.resize(label, (self.w, self.h), interpolation = cv2.INTER_NEAREST) # type: ignore
        else:
            label = cv2.resize(label, (self.w, self.h), interpolation = cv2.INTER_NEAREST)/255 # type: ignore

        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label_s = torch.from_numpy(label.astype(np.uint8)).long()

        label_c = torch.zeros((3, ))
        for i in range(3):
            if torch.sum(label_s == i + 1) > 0:
                label_c[i] = 1

        return image, label_s, label_c, name

class ISIC_Dataset(Dataset):
    def __init__(self, root, list_name, train_num, image_size, dataset, transform=None):
        self.root = root
        self.list_path = self.root + list_name
        self.h, self.w = image_size, image_size
        self.dataset = dataset
        self.df = pd.read_csv(self.list_path.replace('txt', 'csv'))
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.transform = transform
        self.files = []

        for name in self.img_ids:
            if self.dataset == 'skin':
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            elif self.dataset == 'polyp' or self.dataset == 'eyes':
                img_file = os.path.join(self.root, "images/%s.png" % name)
            else:
                img_file = os.path.join(self.root, "images/%s.jpg" % name)

            label_file = os.path.join(self.root, f"masks/{name}_segmentation.png")
            cls_id = self.df['image_id'].tolist().index(name)
            me = self.df._get_value(cls_id, 'melanoma') # type: ignore
            se = self.df._get_value(cls_id, 'seborrheic_keratosis') # type: ignore
            ne = 1 - max(me, se)
            cls_label = np.asarray([me, se, ne])
            self.files.append({"img": img_file, "label": label_file, "name": name, "cls_label": cls_label})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles['img'], cv2.IMREAD_COLOR) # type: ignore
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
        label = cv2.imread(datafiles['label'], cv2.IMREAD_GRAYSCALE) # type: ignore

        if self.transform is not None:
            sample = self.transform(image=image, mask=label)
            image, label = sample['image'], sample['mask']

        image = cv2.resize(image, (self.w, self.h), interpolation = cv2.INTER_LINEAR)
        if self.dataset == 'idrid' or self.dataset == 'eyes':
            label = cv2.resize(label, (self.w, self.h), interpolation = cv2.INTER_LINEAR) # type: ignore
        else:
            label = cv2.resize(label, (self.w, self.h), interpolation = cv2.INTER_LINEAR) / 255.0 # type: ignore

        name = datafiles["name"]
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label_s = torch.from_numpy(label.astype(np.uint8)).long()
        label_c = datafiles['cls_label']

        return image, label_s, label_c, name
