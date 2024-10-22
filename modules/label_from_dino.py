import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
import datetime
import os
import random
import torch.nn.functional as F
import imageio
import joblib

from collections import OrderedDict
from tqdm import tqdm
from torch import multiprocessing
from utils import dcrf
from torch.utils.data import DataLoader

from datasets import voc
from modules.model_attn_aff import WeTr
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
from config import get_config
from utils import evaluate, train_helper, cam_helper, AverageMeter, imutils, dcrf

from modules.PAR import PAR

from dino import vision_transformer_v1 as vits


def disperse_v8(feats, index, cls_token, args):
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()
    _A = A[index, :]

    a = (cls_token @ feats.transpose(1, 2)).squeeze()

    # print(torch.min(a), torch.max(a))
    _a = a > args.thres_cls_token

    _A = _A > args.thres_patch_token

    dis_res = _A.sum(0) * _a

    return dis_res


def disperse_bg(feats, index, bg_thres):
    # Compute the similarity
    A = (feats @ feats.transpose(1, 2)).squeeze()

    _A = A[index, :]

    # _A = torch.tanh(_A)

    _A = _A > bg_thres  # 可设置超参数

    dis_res = _A.sum(0)

    return dis_res


def get_pseudo_labels_rv_cam_10_4(model_dino, inputs, cls, cams_list, reps, refine_mod, scales, img_box, memobank, args, num_class=21):
    # use half precision
    model_dino.half().eval()

    with torch.no_grad():
        num = args.samples_per_gpu
        ps = args.patch_size
        _, _, h, w = inputs.shape
        labels = cls
        labels = torch.cat([torch.ones(args.samples_per_gpu, 1).to(inputs.device), labels], dim=1)
        final_res = []  # Final segmentation results

        _bg_cam = torch.zeros(args.samples_per_gpu, 1, args.crop_size, args.crop_size).to(inputs.device).half()
        _bg_cam[:, 0, :, :] = 0.45
        cams_t = torch.cat([_bg_cam, cams_list], dim=1)
        cams_t[:, 1:, ...] = cams_t[:, 1:, ...] * cls.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, args.crop_size, args.crop_size)

        inputs = inputs.half()

        for i in range(num):  # the i-th image in the batch
            res_list = []
            target_h, target_w = int(h / ps), int(w / ps)  # scale = 1
            _inputs = inputs[i].unsqueeze(0)

            reps_i = F.interpolate(reps[i].unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)
            reps_i_flat = reps_i.reshape(1, 512, -1)  # torch.Size([1, 512, 3136])

            cls_valid_num = len(torch.nonzero(labels[i]))

            for sc_idx, sc in enumerate(scales):
                _inputs = F.interpolate(_inputs, size=(int(sc * h), int(sc * w)), mode='bilinear', align_corners=False)
                _, _, _h, _w = _inputs.shape

                cams = cams_list[i].unsqueeze(0).half()  # torch.Size([1, 20, 64, 64])
                _bg_cam = torch.zeros(1, 1, h, w).to(inputs.device).half()
                _bg_cam[:, 0, :, :] = 1.0 - torch.max(cams[:, 0:, :, :], dim=1)[0]

                # remove crop region in the bg
                bg_cam = torch.zeros_like(_bg_cam) * 255
                coord = img_box[i]
                bg_cam[0, 0, coord[0]:coord[1], coord[2]:coord[3]] = _bg_cam[0, 0, coord[0]:coord[1], coord[2]:coord[3]]
                cams = torch.cat([bg_cam, cams], dim=1)

                # Down sampling
                cams = F.interpolate(cams, size=(int(_h / ps), int(_w / ps)), mode='bilinear', align_corners=False)

                # DINO inference
                _, _, qkv = model_dino.get_intermediate_feat(_inputs, n=1)
                qkv = qkv[0]  # torch.Size([3, 1, 6, 4097, 64])

                # CAMs Index
                cams_t_s = F.interpolate(
                    cams_t, size=(int(_h / ps), int(_w / ps)), mode='bilinear', align_corners=False
                )
                cams_max_index = torch.max(cams_t_s, dim=1)[1]

                q, k, v = qkv[0], qkv[1], qkv[2]
                k = k.transpose(1, 2).reshape(1, int(_h / ps) * int(_w / ps) + 1, -1)
                q = q.transpose(1, 2).reshape(1, int(_h / ps) * int(_w / ps) + 1, -1)
                feats = k[:, 1:, :]
                feats = F.normalize(feats, dim=-1, p=2)

                cls_token = q[:, 0, :]
                cls_token = F.normalize(cls_token, dim=-1, p=2)

                # Generate affinity map
                for c in (torch.nonzero(labels[i])):
                    _c = c[0]

                    # cams[0][_c][cams[0][_c] < 0.25] = 0
                    # large_num = len(torch.nonzero(cams[0][_c]))
                    large_num = len(torch.nonzero(cams[0][_c] * (cams_max_index[i] == _c)))

                    if _c != 0:  # non-background class
                        k = round(large_num * args.topk) + 1
                        s = k
                    else:  # background class
                        k = round(large_num * args.topk) + 1
                        s = k

                    # _, anchor_index = torch.topk(cams[0][_c].view(-1), k=k)
                    _, anchor_index = torch.topk((cams[0][_c] * (cams_max_index[i] == c)).view(-1), k=k)
                    sampled_indices = torch.randperm(k)[:s]
                    anchor_index = anchor_index[sampled_indices]

                    if _c != 0:
                        res = disperse_v8(
                            feats=feats, index=anchor_index, cls_token=cls_token, args=args
                        ).reshape(int(_h / ps), int(_w / ps)).unsqueeze(0).unsqueeze(0).type(torch.float)
                    else:
                        res = disperse_bg(
                            feats=feats, index=anchor_index, bg_thres=args.thres_bg_token
                        ).reshape(int(_h / ps), int(_w / ps)).unsqueeze(0).unsqueeze(0).type(torch.float)

                    res = res + F.adaptive_max_pool2d(-res, (1, 1))
                    res /= F.adaptive_max_pool2d(res, (1, 1)) + 1e-5

                    res = F.interpolate(res, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    res_list.append(res)

            # Affinity map fusion
            _cls = torch.nonzero(labels[i])
            final_res_i = torch.zeros(num_class, target_h, target_w, device=inputs.device)
            for j, c in enumerate(_cls):  # the c-th class
                tmp = []
                for k in range(len(scales)):
                    tmp.append(res_list[j + k * len(_cls)])

                a = torch.sum(torch.stack(tmp, dim=0), dim=0)
                a = a + F.adaptive_max_pool2d(-a, (1, 1))
                a /= F.adaptive_max_pool2d(a, (1, 1)) + 1e-5

                if c != 0:
                    memo_rep = memobank[c - 1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, target_h, target_w)
                    memo_rep = F.normalize(memo_rep, dim=1, p=2)
                    reps_i = F.normalize(reps_i, dim=1, p=2)

                    if cls_valid_num > 2:
                        sim = torch.einsum('nchw,nchw->nhw', memo_rep, reps_i)

                        sim[sim < 0] = 0
                        sim = sim ** 1.5
                        sim = sim + F.adaptive_max_pool2d(-sim, (1, 1))
                        sim /= F.adaptive_max_pool2d(sim, (1, 1)) + 1e-5

                        a = sim * a

                    _, anchor_reps_idx = torch.topk(a.clone().detach().view(-1), k=8)
                    reps_i_c = reps_i_flat.permute(0, 2, 1)[:, anchor_reps_idx, :][0].mean(0)
                    memobank[c - 1] = args.momentum * memobank[c - 1] + (1 - args.momentum) * reps_i_c

                else:

                    a[a < args.thres_bg_aff_l] = args.thres_bg_aff_l  # Set threshold of background class
                    a[a > args.thres_bg_aff_h] = args.thres_bg_aff_h  # Set threshold of background class

                final_res_i[c] = a

            final_res.append(final_res_i)

        # Merge results
        final_res = torch.stack(final_res, dim=0)

        # Segmentation
        # ----------------------------------------------------------------
        down_scale = 2

        final_res = F.interpolate(
            final_res, size=(h // down_scale, w // down_scale), mode='bilinear', align_corners=False
        ).squeeze(0)

        # Refinement
        _inputs = F.interpolate(inputs, size=(h // down_scale, w // down_scale), mode='bilinear', align_corners=False)
        input_res = imutils.denormalize_img2(_inputs)
        _refined_cams = refine_mod(input_res, final_res)
        _refined_cams = F.interpolate(_refined_cams, size=(h, w), mode='bilinear', align_corners=False)
        _pseudo_label = torch.argmax(_refined_cams, dim=1)

        # output
        pseudo_label = torch.ones_like(_pseudo_label) * 255
        for idx, coord in enumerate(img_box):
            pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label
