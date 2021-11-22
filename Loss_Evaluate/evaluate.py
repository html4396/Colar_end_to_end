from tools import utils
import torch
import torch.nn.functional as F
from torch import nn

all_class_name = [
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking"]


def RoTAD_evaluate(results, command, log_file):
    map, aps, cap, caps = utils.frame_level_map_n_cap(results)
    out = ' [IDU-{}] mAP: {:.4f}\n'.format(command, map)
    print(out)
    with open(log_file, 'a+') as f:
        f.writelines(out)

    for i, ap in enumerate(aps):
        cls_name = all_class_name[i]
        out = '{}: {:.4f}\n'.format(cls_name, ap)

        with open(log_file, 'a+') as f:
            f.writelines(out)


class SetCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.criterion_CE = nn.CrossEntropyLoss(reduction=reduction)
        self.criterion_MSE = nn.MSELoss()
        self.criterion_KL = nn.KLDivLoss()

    def forward(self, outputs, targets, type):
        if type == 'CE':
            loss = self.criterion_CE(outputs, targets)
        elif type == 'MSE':
            outputs = F.softmax(outputs, dim=1)
            targets = F.softmax(targets, dim=1)
            loss = self.criterion_MSE(outputs, targets)
        elif type == 'KL':
            outputs = F.softmax(outputs, dim=1)
            targets = F.softmax(targets, dim=1)
            loss1 = self.criterion_KL(outputs, targets)
            loss2 = self.criterion_KL(targets, outputs)
            loss = loss1 + loss2
        return loss
