import torch.nn as nn
import torch
import torch.nn.functional as F


class RoTal_dynamic(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(RoTal_dynamic, self).__init__()
        self.channels = 512
        self.conv1_3_k = nn.Conv1d(ch_in, self.channels, kernel_size=3, stride=1, padding=1)
        self.conv1_3_v = nn.Conv1d(ch_in, self.channels, kernel_size=3, stride=1, padding=1)
        self.conv1_feature = nn.Conv1d(ch_in, self.channels, kernel_size=3, stride=1, padding=1)

        self.conv2_3_k = nn.Conv1d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.conv2_3_v = nn.Conv1d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv1d(self.channels, ch_out, kernel_size=1, stride=1, padding=0)

        self.opt = nn.ReLU()

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = F.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight

    def sum(self, value, y_weight):
        y_weight = y_weight.permute(0, 2, 1)
        y_sum = torch.bmm(value, y_weight)
        sum = value[:, :, -1:] + y_sum
        return torch.cat((value[:, :, :-1], sum), dim=-1)

    def forward(self, input):
        k = self.conv1_3_k(input)
        v = self.conv1_3_v(input)
        y_weight = self.weight(k, k[:, :, -1:])
        feat1 = self.sum(v, y_weight)
        feat1 = self.opt(feat1)

        k = self.conv2_3_k(feat1)
        v = self.conv2_3_v(feat1)
        y_weight = self.weight(k, k[:, :, -1:])
        feat2 = self.sum(v, y_weight)
        feat2 = self.opt(feat2)

        return self.conv3_1(feat2)


if __name__ == '__main__':
    model_dynamic = RoTal_dynamic(2048, 22)
    a = torch.load('/disk/sunyuan/1105/new_online /online.pth')['model']
    model_dynamic.load_state_dict(torch.load('/disk/sunyuan/1105/new_online /online.pth')['model'])
    model = model_dynamic.cuda()
    pass
