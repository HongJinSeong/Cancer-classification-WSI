import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import *
from torch.nn.parameter import Parameter

import torch
import torch.nn.functional as F
import torch.nn as nn
class ensemble_M(nn.Module):
    def __init__(self,model, weights, weight_path):
        super(ensemble_M, self).__init__()
        self.model = model
        self.weigths = weights
        self.weight_path = weight_path

    ## ENSEMBLE 선택지 싹다 평균 or 싹다 voting
    def forward(self, x,tab):
        vote = []
        score = []

        for wp in self.weigths:
            self.model.load_state_dict(torch.load(self.weight_path + wp))
            output, _, _ = self.model(x,tab)
            score.append(output.detach().cpu().numpy().tolist()[0][0])
            vote.append(torch.where(output >= 0.5, 1, 0).detach().cpu().numpy().tolist()[0][0])
        return score, vote

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(p={self.p.data.tolist()[0]:.4f}, eps={str(self.eps)})"
        )

class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch // r)
        self.linear_2 = nn.Linear(in_ch // r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)

        x = input_x * x

        return x

class MIL_pretrained_CNN(nn.Module):
    def __init__(self, p_model):
        super(MIL_pretrained_CNN, self).__init__()
        self.feature_extractor_part1 = p_model

        self.GAP = torch.nn.AdaptiveAvgPool2d((4,1))

        self.gempool = GeM()

        self.SEB_LIN = SEBlock(768)

        self.Classifier = nn.Linear(768, 1)

        self.sig = nn.Sigmoid()


    def forward(self, x,tab):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)

        H = H['x1'].permute(1,0,2,3).view(768,40*4,4).unsqueeze(0)

        # H = self.GAP(H)

        H = self.gempool(H)

        H = H.flatten(1)

        H = self.SEB_LIN(H)

        output = self.sig(self.Classifier(H))
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.ReLU()
        )
        self.gempool = GeM()

        self.SEB_LIN = SEBlock(1024)

        self.Classifier = nn.Linear(1024, 1)

    def forward(self, x):
        H = self.feature_extractor_part1(x)

        H = self.gempool(H)

        H = H.flatten(1)

        H = self.SEB_LIN(H)

        output = self.Classifier(H)

        return output

class MIL_Attention(nn.Module):
    def __init__(self):
        super(MIL_Attention, self).__init__()
        self.L = 256  # 512 node fully connected layer
        self.D = 128  # 128 node attention layer
        self.K = 1
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(32 * 29 * 29, self.L),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout()
        )

        # self.tabular_extractor = custom_tab_v2()

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x,tab):
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 32 * 29 * 29)
        H = self.feature_extractor_part2(H)

        # T = self.tabular_extractor(tab)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)
        # The probability that a given bag is malignant or benign
        # image feature + tabular feature
        # M = torch.cat((M, T), dim=1)
        Y_prob = self.classifier(M)
        # The prediction given the probability (Y_prob >= 0.5 returns a Y_hat of 1 meaning malignant)
        Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, Y_hat, A.byte()


class custom_tab_v2(nn.Module):
    def __init__(self):
        super().__init__()
        emb_in = [4, 3]

        emb_Layers = []
        for ein in emb_in:
            emb_Layers.append(nn.Embedding(ein, ein * 3))

        self.emb_Layers = nn.ModuleList(emb_Layers)

        Lin_Layers = []
        for ein in emb_in:
            Lin_Layers.append(nn.Sequential(nn.Linear(ein * 3, 32), nn.ReLU(inplace=True), nn.Dropout(0.2)))

        self.Lin_Layers = nn.ModuleList(Lin_Layers)

        self.cat_emLIN = nn.Linear(64, 512)
        self.cat_em_Re = nn.ReLU(inplace=True)
        self.cat_emdrop = nn.Dropout(0.1)

        self.numericLIN = nn.Linear(13, 512)
        self.numeric_Re = nn.ReLU(inplace=True)
        self.numericdrop = nn.Dropout(0.1)

        self.tab_fc = nn.Linear(1024, 768)
        self.tab_Re = nn.ReLU(inplace=True)
        self.tab_drop = nn.Dropout(0.1)

        self.tab_fc1 = nn.Linear(768, 512)
        self.tab_Re1 = nn.ReLU(inplace=True)

    def forward(self, tarb):

        emb_output = self.Lin_Layers[0](self.emb_Layers[0](tarb[:, 0].type(torch.int64)))

        for e_idx in range(1, len(self.emb_Layers)):
            emb_output = torch.cat(
                (emb_output, self.Lin_Layers[e_idx](self.emb_Layers[e_idx](tarb[:, e_idx].type(torch.int64)))), dim=1)

        emb_output = self.cat_emdrop(self.cat_em_Re(self.cat_emLIN(emb_output)))

        numeric_output = self.numericdrop(self.numeric_Re(self.numericLIN(tarb[:, 2:].type(torch.float))))

        tab_feature = torch.cat((emb_output, numeric_output), dim=1)

        tab_feature = self.tab_drop(self.tab_Re(self.tab_fc(tab_feature)))

        tab_feature = self.tab_Re1(self.tab_fc1(tab_feature))

        return tab_feature

class custom_multimodal_v2(nn.Module):
    def __init__(self, CNNmodel,tabmodel):
        super().__init__()
        self.CNNmodel = CNNmodel
        self.CNNmodel.fc = nn.Identity()
        self.tabmodel = tabmodel
        self.tabmodel.fc = nn.Identity()

        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 1)


    def forward(self, imgs, tarb):
        img_feature = self.CNNmodel(imgs)
        tab_feature = self.tabmodel(tarb)

        fusion_feature = torch.cat((img_feature,tab_feature), dim = 1)

        outputs = self.fc2(self.relu1(self.fc1(fusion_feature)))

        return outputs


####region semantic segmentation
# PyTorch
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

####endregion