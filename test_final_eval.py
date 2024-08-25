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

from datasets import voc
from modules.model_attn_aff import WeTr
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
from config import get_config
from utils import evaluate

import warnings

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_set = 'val'


def validate(model, dataset, test_scales=None, args=None):
    _preds, _gts, _msc_preds = [], [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, _ = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            #######
            # resize long side to 512
            _, _, h, w = inputs.shape
            ratio = 512 / max(h, w)  # args.resize_long
            _h, _w = int(h * ratio), int(w * ratio)
            inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
            #######

            segs_list = []
            inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
            _, segs_cat, _, _, _, _ = model(inputs_cat, )
            segs = segs_cat[0].unsqueeze(0)

            _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
            segs_list.append(_segs)

            _, _, h, w = segs_cat.shape

            for s in test_scales:
                if s != 1.0:
                    _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                    inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                    _, segs_cat, _, _, _, _ = model(inputs_cat, )

                    _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                    _segs = (_segs_cat[0, ...] + _segs_cat[1, ...].flip(-1)) / 2
                    segs_list.append(_segs)

            msc_segs = torch.max(torch.stack(segs_list, dim=0), dim=0)[0].unsqueeze(0)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_preds = torch.argmax(resized_segs, dim=1)

            resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

            _preds += list(seg_preds.cpu().numpy().astype(np.int16))
            _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
            _gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.work_dir + '/logit/' + name[0] + '.npy',
                    {"segs": segs.cpu().numpy(), "msc_segs": msc_segs.cpu().numpy()})

    return _gts, _preds, _msc_preds


def main(model, args):
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.root_dir, name_list_dir=args.name_list_dir, split=eval_set,
        stage='val', aug=False
    )

    model.eval()

    gts, preds, msc_preds = validate(model=model, dataset=val_dataset, test_scales=[1, 0.75, 1.25], args=args)
    torch.cuda.empty_cache()

    seg_score = evaluate.scores(gts, preds)
    msc_seg_score = evaluate.scores(gts, msc_preds)

    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    return seg_score, msc_seg_score


def final_eval(model, args):
    args.work_dir = os.path.join(args.work_dir, eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    seg_score, msc_seg_score = main(model, args)

    return seg_score, msc_seg_score
