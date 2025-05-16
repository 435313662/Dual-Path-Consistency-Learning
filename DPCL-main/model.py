import torch
import torch.nn as nn
from loss import LossFunction
import torchvision.transforms as transforms


class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(EnhanceNetwork, self).__init__()

        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)

        illu = fea + input
        illu = torch.clamp(illu, 0.1, 0.6)#0.1 0.9

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        delta = input - fea

        return delta



class Network(nn.Module,):

    def __init__(self, stage=3,isRandom=False):
        super(Network, self).__init__()
        self.transform= transforms.Compose([transforms.ColorJitter(contrast=(0.6,1.4), saturation=(0.6,1.4),brightness=(0.6,1.4))])#0.6,1.4
        self.stage = stage
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.isRandom = isRandom
        self.calibrate = CalibrateNetwork(layers=3, channels=16) if not isRandom else None
        self._criterion = LossFunction(self.isRandom)
        self.nnloss= nn.MSELoss()
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        if self.isRandom==False:
            ilist, rlist, inlist, attlist = [], [], [], []
            input_op = input
            for i in range(self.stage):
                inlist.append(input_op)
                i = self.enhance(input_op)
                r = input / i
                r = torch.clamp(r, 0, 1)
                att = self.calibrate(r)
                input_op = input + att
                ilist.append(i)
                rlist.append(r)
                attlist.append(torch.abs(att))
        else:
            ilist, rlist, inlist, attlist = [], [], [], []
            input_random= self.transform(input)
            inlist.append(input)
            inlist.append(input_random)
            i_ori= self.enhance(input)
            i_random= self.enhance(input_random)
            r_ori = input / i_ori
            r_random = input_random / i_random
            r_ori = torch.clamp(r_ori, 0, 1)
            r_random = torch.clamp(r_random, 0, 1)
            ilist.append(i_ori)
            ilist.append(i_random)
            rlist.append(r_ori)
            rlist.append(r_random)

        return ilist, rlist, inlist, attlist

    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(len(i_list)):
            loss += self._criterion(in_list[i], i_list[i])
        if self.isRandom:
            loss+=0.5*self.nnloss(i_list[0], i_list[1])
        return loss



class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

