import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from . import mix_transformer
import numpy as np


class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained_dir=None, pooling=None, ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained_dir:
            state_dict = torch.load(pretrained_dir)
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.decoder = conv_head.LargeFOV(self.in_channels[-1], out_planes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes - 1, kernel_size=1,
                                    bias=False)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x, cam_only=False, cls_only=False, cams_only_with_grad=False, seg_detach=True):

        _x, _attns = self.encoder(x)

        _x1, _x2, _x3, _x4 = _x

        # print(_x1.shape, _x2.shape, _x3.shape, _x4.shape)

        if cls_only:
            cls_x4 = self.pooling(_x4, (1, 1))
            cls_x4 = self.classifier(cls_x4)
            cls_x4 = cls_x4.view(-1, self.num_classes - 1)
            return cls_x4

        seg = self.decoder(_x)
        # seg = self.decoder(_x4)

        attn_cat = torch.cat(_attns[-2:], dim=1)  # .detach()

        # print('???', _attns[-1][0])

        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)  # Equation 3

        attn_pred = self.attn_proj(attn_cat)  # 每一层都是 8 个 head，取后面两层总共 16 个 MHSA
        attn_pred = torch.sigmoid(attn_pred)[:, 0, ...]  # squeeze

        if cam_only:
            cam_s4 = F.conv2d(_x4, self.classifier.weight).detach()
            return cam_s4, attn_pred

        if cams_only_with_grad:
            cam_s4 = F.conv2d(_x4, self.classifier.weight)
            return cam_s4

        # _x4 = self.dropout(_x4.clone()
        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        cam_s4 = F.conv2d(_x4, self.classifier.weight)

        # attns = [attn[:,0,...] for attn in _attns]
        # attns.append(attn_pred)
        return cls_x4, seg, _attns, attn_pred, _x4, cam_s4


if __name__ == "__main__":
    wetr = WeTr(
        'mit_b1', num_classes=20, embedding_dim=256, stride=[4, 2, 2, 1],
        pretrained_dir='/media/store1/wyc/exps/T_Wseg/pretrained/mit_b1.pth', pooling='gmp'
    ).cuda()
    # wetr._param_groups()
    dummy_input = torch.rand(2, 3, 512, 512).cuda()
    print(wetr(dummy_input)[1].shape)