import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes=5):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        tp = torch.sum(score * target)
        fp = torch.sum(score) - tp
        fn = torch.sum(target) - tp
        loss = (2*tp + smooth)/(2*tp + fn + fp + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, n_classes=5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            temp_prob = torch.unsqueeze(temp_prob, 1)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _focal_loss(self, input, target):
        target = target.float()

        input = input.clamp(self.eps, 1 - self.eps)
        loss = - (target * torch.pow((1 - input), self.gamma) * torch.log(input) +
                  (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input))
        return loss

    def forward(self, inputs, targets):
        targets = self._one_hot_encoder(targets)
        assert inputs.size() == targets.size(), 'predict & target shape do not match'
        loss = 0.0
        for i in range(0, self.n_classes):
            focal = self._focal_loss(inputs[:, i], targets[:, i])
            loss += focal.mean()

        return loss / self.n_classes

class Dice_and_FocalLoss(nn.Module):
    def __init__(self, gamma=2, n_classes=5):
        super(Dice_and_FocalLoss, self).__init__()
        self.dice_loss = DiceLoss(n_classes)
        self.focal_loss = FocalLoss(gamma, n_classes)

    def forward(self, input, target):
        loss = self.dice_loss(input, target) + self.focal_loss(input, target)

        return loss
