import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import datetime

from tqdm import tqdm

from utils import cam_helper
from .optimizer import PolyWarmupAdamW
from utils.AverageMeter import AverageMeter
from utils import evaluate, corrloss, imutils

import sys

sys.path.append("./bilateralfilter/build/lib.linux-x86_64-3.6")
from bilateralfilter import bilateralfilter, bilateralfilter_batch


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    # _hw = (h + max(dilations)) * (w + max(dilations))
    mask = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius + 1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius + 1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def get_optimizer(param_groups, learning_rate, weight_decay, warmup_iter, max_iters, warmup_ratio, power):
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": learning_rate * 10,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": learning_rate * 10,
                "weight_decay": weight_decay,
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=[0.9, 0.999],
        warmup_iter=warmup_iter,
        max_iter=max_iters,
        warmup_ratio=warmup_ratio,
        power=power
    )
    return optimizer


def get_optimizer_v10_8(param_groups, learning_rate, weight_decay, warmup_iter, max_iters, warmup_ratio, power):
    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups[2],
                "lr": learning_rate * 10,
                "weight_decay": weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": learning_rate * 10,
                "weight_decay": weight_decay,
            },
        ],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=[0.9, 0.999],
        warmup_iter=warmup_iter,
        max_iter=max_iters,
        warmup_ratio=warmup_ratio,
        power=power
    )
    return optimizer


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_seg_loss_v9(pred, label, ignore_index=255):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    # bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    bg_loss = ce(pred, bg_label.type(torch.long)).sum() / (bg_sum + 1e-6)

    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    # fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    fg_loss = ce(pred, fg_label.type(torch.long)).sum() / (fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5


def get_seg_loss_with_grad_clip(pred, label, loss_thres, margin=0.5, ignore_index=255, args=None):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    # Background
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    bg_loss = ce(pred, bg_label.type(torch.long))

    for i in range(args.samples_per_gpu):
        clip_mask_i = (bg_loss[i] > (loss_thres + margin))
        bg_loss[i][clip_mask_i] = 0
        bg_sum -= clip_mask_i.long().sum()

    bg_loss = bg_loss.sum() / (bg_sum + 1e-6)

    # Foreground
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    fg_loss = ce(pred, fg_label.type(torch.long))

    # for i in range(args.samples_per_gpu):
    #     clip_mask_i = (fg_loss[i] > (loss_thres + margin))
    #     fg_loss[i][clip_mask_i] = 0
    #     fg_sum -= clip_mask_i.long().sum()

    fg_loss = fg_loss.sum() / (fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5


def get_seg_loss_with_grad_clip_v2(pred, label, loss_thres, margin=0.5, ignore_index=255, args=None):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    # Background
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    bg_loss = ce(pred, bg_label.type(torch.long))

    for i in range(args.samples_per_gpu):
        clip_mask_i = (bg_loss[i] > (loss_thres + margin))
        bg_loss[i][clip_mask_i] = (loss_thres + margin)  # clip

    bg_loss = bg_loss.sum() / (bg_sum + 1e-6)

    # Foreground
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    fg_loss = ce(pred, fg_label.type(torch.long))

    # for i in range(args.samples_per_gpu):
    #     clip_mask_i = (fg_loss[i] > (loss_thres + margin))
    #     fg_loss[i][clip_mask_i] = 0
    #     fg_sum -= clip_mask_i.long().sum()

    fg_loss = fg_loss.sum() / (fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5, bg_loss.item()


def get_seg_loss_with_grad_clip_v3(pred, label, loss_thres, margin=0.5, ignore_index=255, args=None):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    # Background
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    bg_loss = ce(pred, bg_label.type(torch.long))

    # for i in range(args.samples_per_gpu):
    #     clip_mask_i = (bg_loss[i] > (loss_thres + margin))
    #     bg_loss[i][clip_mask_i] = (loss_thres + margin)  # clip

    bg_loss = bg_loss.sum() / (bg_sum + 1e-6)

    # Foreground
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_sum = (fg_label != ignore_index).long().sum()
    fg_loss = ce(pred, fg_label.type(torch.long))

    for i in range(args.samples_per_gpu):
        clip_mask_i = (fg_loss[i] > (loss_thres + margin))
        fg_loss[i][clip_mask_i] = (loss_thres + margin)  # clip

    fg_loss = fg_loss.sum() / (fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5, fg_loss.item()


def get_seg_loss_with_grad_clip_v4(pred, label, loss_thres, margin=1.0, ignore_index=255, args=None):
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    # Background
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_sum = (bg_label != ignore_index).long().sum()
    bg_loss = ce(pred, bg_label.type(torch.long))

    # for i in range(args.samples_per_gpu):
    #     clip_mask_i = (bg_loss[i] > (loss_thres + margin))
    #     bg_loss[i][clip_mask_i] = (loss_thres + margin)  # clip

    bg_loss = bg_loss.sum() / (bg_sum + 1e-6)

    # Foreground
    fg_loss = torch.zeros(1).to(pred.device)
    fg_sum = 0
    for i in range(args.samples_per_gpu):
        label_i = label[i].clone().unsqueeze(0)
        cls_label_i = torch.unique(label_i)

        for c in cls_label_i:
            if c != 0 and c != ignore_index:
                fg_label_i_c = label_i.clone()
                fg_label_i_c[label_i != c] = ignore_index
                fg_sum_i_c = (fg_label_i_c == c).long().sum()
                fg_sum += fg_sum_i_c

                fg_loss_i_c = ce(pred[i].unsqueeze(0), fg_label_i_c.type(torch.long))
                fg_loss_i_c_no_clip_mean = fg_loss_i_c.sum().item() / fg_sum_i_c  # sum loss

                # clip
                thres = max(loss_thres[c - 1], fg_loss_i_c_no_clip_mean) + margin
                clip_mask_i_c = (fg_loss_i_c > thres)
                fg_loss_i_c[clip_mask_i_c] = thres  # clip
                fg_loss_i_c = fg_loss_i_c.sum()  # sum loss

                fg_loss += fg_loss_i_c

                loss_thres[c - 1] = 0.9 * loss_thres[c - 1] + 0.1 * (fg_loss_i_c_no_clip_mean / fg_sum_i_c)

    fg_loss = fg_loss / (fg_sum + 1e-6)

    return (bg_loss + fg_loss) * 0.5


def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:, 0, ...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
    _img[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
    _img[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()


def max_norm(p, version='torch', e=1e-5):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
            min_v = torch.min(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
            p = F.relu(p - min_v - e) / (max_v - min_v + e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
            min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
            p = F.relu(p - min_v - e) / (max_v - min_v + e)
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p < 0] = 0
            max_v = np.max(p, (1, 2), keepdims=True)
            min_v = np.min(p, (1, 2), keepdims=True)
            p[p < min_v + e] = 0
            p = (p - min_v - e) / (max_v + e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p < 0] = 0
            max_v = np.max(p, (2, 3), keepdims=True)
            min_v = np.min(p, (2, 3), keepdims=True)
            p[p < min_v + e] = 0
            p = (p - min_v - e) / (max_v + e)
    return p


def validate(model=None, data_loader=None, cfg=None):
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, _, attn_pred = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    model.train()
    return cls_score, seg_score, cam_score


def validate_v6(model=None, data_loader=None, cfg=None):
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, _, attn_pred, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    model.train()
    return cls_score, seg_score, cam_score


def validate_v8(model=None, data_loader=None, cfg=None):
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, _, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    model.train()
    return cls_score, seg_score, cam_score


def validate_v86(model=None, data_loader=None, cfg=None):
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, _, _, _, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    model.train()
    return cls_score, seg_score, cam_score


def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


def validate_v811(model=None, data_loader=None, cfg=None):
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()

            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, attns, attn_pred, reps, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=cfg.radius)
            valid_cam_resized = F.interpolate(resized_cam, size=(H, W), mode='bilinear', align_corners=False)
            aff_cam = cam_helper.propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask,
                                                           cls_labels=cls_label,
                                                           bkg_score=0.35)
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            aff_label = aff_cam.argmax(dim=1)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            aff_gts += list(aff_label.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    aff_score = evaluate.scores(gts, aff_gts)
    model.train()
    return cls_score, seg_score, cam_score, aff_score


def validate_coco(model=None, data_loader=None, cfg=None):
    preds, gts, cams, aff_gts = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()

            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, attns, attn_pred, reps, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=cfg.radius)
            valid_cam_resized = F.interpolate(resized_cam, size=(H, W), mode='bilinear', align_corners=False)
            aff_cam = cam_helper.propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask,
                                                           cls_labels=cls_label,
                                                           bkg_score=0.35)
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            aff_label = aff_cam.argmax(dim=1)

            preds.append(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int8))
            cams.append(cam_label.cpu().numpy().astype(np.int8))
            gts.append(labels.cpu().numpy().astype(np.int8))
            aff_gts.append(aff_label.cpu().numpy().astype(np.int8))

            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds, num_classes=81)
    cam_score = evaluate.scores(gts, cams, num_classes=81)
    aff_score = evaluate.scores(gts, aff_gts, num_classes=81)

    model.train()
    return cls_score, seg_score, cam_score, aff_score


def validate_v10_6(model=None, data_loader=None, cfg=None):
    preds, aux_pred, gts, cams, aff_gts = [], [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs, aux_segs, attns, attn_pred, reps, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_aux_segs = F.interpolate(aux_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = get_mask_by_radius(h=H, w=W, radius=cfg.radius)
            valid_cam_resized = F.interpolate(resized_cam, size=(H, W), mode='bilinear', align_corners=False)
            aff_cam = cam_helper.propagte_aff_cam_with_bkg(valid_cam_resized, aff=attn_pred, mask=infer_mask,
                                                           cls_labels=cls_label,
                                                           bkg_score=0.35)
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            aff_label = aff_cam.argmax(dim=1)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            aux_pred += list(torch.argmax(resized_aux_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            aff_gts += list(aff_label.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    aux_seg_score = evaluate.scores(gts, aux_pred)
    cam_score = evaluate.scores(gts, cams)
    aff_score = evaluate.scores(gts, aff_gts)
    model.train()
    return cls_score, seg_score, aux_seg_score, cam_score, aff_score


def validate_v9(model=None, data_loader=None, cfg=None):
    preds1, preds2, gts, cams, aff_gts = [], [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            cls, segs1, segs2, _, _, _, _ = model(inputs, )

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs1 = F.interpolate(segs1, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_segs2 = F.interpolate(segs2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            ###
            _cams = cam_helper.multi_scale_cam(model, inputs, [1, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_helper.cam_to_label(resized_cam, cls_label, args=cfg)

            preds1 += list(torch.argmax(resized_segs1, dim=1).cpu().numpy().astype(np.int16))
            preds2 += list(torch.argmax(resized_segs2, dim=1).cpu().numpy().astype(np.int16))

            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            valid_label = torch.nonzero(cls_label[0])[:, 0]
            out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score_d = evaluate.scores(gts, preds1)
    seg_score_c = evaluate.scores(gts, preds2)
    cam_score = evaluate.scores(gts, cams)
    model.train()
    return cls_score, seg_score_d, seg_score_c, cam_score


def update_ema_variables(model, ema_model, alpha, global_step):
    with torch.no_grad():
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_((1 - alpha) * param.detach().data)


def get_aux_loss(inputs, targets):
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    # inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def get_aff_loss(inputs, targets):
    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    # inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count


def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375]):
    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:, 0, ...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
    _img[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
    _img[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()


class DenseEnergyLossFunction(Function):
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs

        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)

        # averaged by the number of images
        densecrf_loss /= ctx.N

        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2 * grad_output * torch.from_numpy(ctx.AS) / ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None


class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images, scale_factor=self.scale_factor)
        scaled_segs = F.interpolate(segmentations, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1), scale_factor=self.scale_factor).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label, scale_factor=self.scale_factor, mode='nearest')
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight * DenseEnergyLossFunction.apply(
            scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy * self.scale_factor, scaled_ROIs, unlabel_region)

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    # time_now = datetime.datetime.strptime(time_now.strftime('%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S')

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


class ConsistencyLoss(nn.Module):
    def __init__(self, args):
        super(ConsistencyLoss, self).__init__()
        self.corr_loss = corrloss.ContrastiveCorrelationLoss()
        self.args = args

    def forward(self, model, local_crops, global_segs, global_cam, crop_num, crop_size):
        local_crops = local_crops.reshape(-1, 3, crop_size, crop_size)  # batch_size * crop_num
        local_crops = imutils.denormalize_img2(local_crops)
        augment = imutils.get_rand_aug()
        local_crops = augment(local_crops)

        # Inference
        _, local_segs, _, _, _ = model(local_crops, seg_detach=self.args.seg_detach)
        local_cam = cam_helper.multi_scale_cam(model, inputs=local_crops, scales=[1])

        local_segs = F.interpolate(local_segs, size=local_cam.shape[2:], mode='bilinear', align_corners=False)
        local_segs = local_segs.reshape(
            self.args.samples_per_gpu, crop_num, 21, local_segs.shape[-2], local_segs.shape[-1]
        )
        local_cam = local_cam.reshape(
            self.args.samples_per_gpu, crop_num, 20, local_cam.shape[-2], local_cam.shape[-1]
        )

        total_loss = 0
        for i in range(self.args.samples_per_gpu):
            i_local_cam = local_cam[i]
            i_local_segs = local_segs[i]
            i_global_cam = global_cam[i].repeat(crop_num, 1, 1, 1)
            i_global_segs = global_segs[i].repeat(crop_num, 1, 1, 1)

            loss_corr = self.corr_loss(i_local_cam, i_global_cam, i_local_segs, i_global_segs)
            total_loss += loss_corr

        total_loss = total_loss / self.args.samples_per_gpu

        return total_loss


def get_masked_ptc_loss(inputs, mask):
    b, c, h, w = inputs.shape

    inputs = inputs.reshape(b, c, h * w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1, 2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5 * (1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum() + 1)) + 0.5 * torch.sum(
        neg_mask * inputs_cos) / (neg_mask.sum() + 1)
    return loss


class AlignmentLoss(nn.Module):
    def __init__(self, args, num_classes=20):
        super(AlignmentLoss, self).__init__()
        self.args = args
        self.num_classes = num_classes

    def forward(self, model, local_crops, local_crops_pseudo_label, down_scale, crop_num, crop_size):
        local_crops = local_crops.reshape(-1, 3, crop_size, crop_size)  # batch_size * crop_num
        local_crops = imutils.denormalize_img2(local_crops)
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(-1, 1, crop_size, crop_size)
        local_crops, local_crops_pseudo_label = imutils.augment_data(local_crops, local_crops_pseudo_label)

        # # 降采样减小计算量
        # local_crops = F.interpolate(
        #     local_crops, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        # )

        # Inference
        _, _, _, _, _, local_cams = model(local_crops, seg_detach=self.args.seg_detach)
        local_cams = F.interpolate(
            local_cams, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        )

        # Reshape to -> (batch_size, crop_num, C, crop_size // down_scale, crop_size // down_scale)
        local_cams = local_cams.reshape(
            -1, crop_num, self.num_classes, crop_size // down_scale, crop_size // down_scale
        )
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(
            -1, crop_num, 1, crop_size, crop_size
        )

        loss_total = 0
        for i in range(self.args.samples_per_gpu):
            local_cams_i = local_cams[i]
            local_crops_pseudo_label_i = F.interpolate(
                local_crops_pseudo_label[i], size=(crop_size // down_scale, crop_size // down_scale), mode='nearest'
            )
            global_segs_i = cam_helper.label_to_aff_mask(local_crops_pseudo_label_i.squeeze(1))
            loss_i = get_masked_ptc_loss(local_cams_i, global_segs_i)

            loss_total += loss_i

        alignment_loss = loss_total / (self.args.samples_per_gpu * crop_num)

        return alignment_loss


class AlignmentLoss_v9(nn.Module):
    def __init__(self, args):
        super(AlignmentLoss_v9, self).__init__()
        self.args = args

    def forward(self, model, local_crops, local_crops_pseudo_label, down_scale, crop_num, crop_size):
        local_crops = local_crops.reshape(-1, 3, crop_size, crop_size)  # batch_size * crop_num
        local_crops = imutils.denormalize_img2(local_crops)
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(-1, 1, crop_size, crop_size)
        local_crops, local_crops_pseudo_label = imutils.augment_data(local_crops, local_crops_pseudo_label)

        # # 降采样减小计算量
        # local_crops = F.interpolate(
        #     local_crops, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        # )

        # Inference
        _, _, _, _, _, _, local_cams = model(local_crops, seg_detach=self.args.seg_detach)
        local_cams = F.interpolate(
            local_cams, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        )

        # Reshape to -> (batch_size, crop_num, C, crop_size // down_scale, crop_size // down_scale)
        local_cams = local_cams.reshape(
            -1, crop_num, 20, crop_size // down_scale, crop_size // down_scale
        )
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(
            -1, crop_num, 1, crop_size, crop_size
        )

        loss_total = 0
        for i in range(self.args.samples_per_gpu):
            local_cams_i = local_cams[i]
            local_crops_pseudo_label_i = F.interpolate(
                local_crops_pseudo_label[i], size=(crop_size // down_scale, crop_size // down_scale), mode='nearest'
            )
            global_segs_i = cam_helper.label_to_aff_mask(local_crops_pseudo_label_i.squeeze(1))
            loss_i = get_masked_ptc_loss(local_cams_i, global_segs_i)

            loss_total += loss_i

        alignment_loss = loss_total / (self.args.samples_per_gpu * crop_num)

        return alignment_loss


class AlignmentLoss_v10(nn.Module):
    def __init__(self, args):
        super(AlignmentLoss_v10, self).__init__()
        self.args = args

    def forward(self, model, local_crops, local_crops_pseudo_label, down_scale, crop_num, crop_size):
        local_crops = local_crops.reshape(-1, 3, crop_size, crop_size)  # batch_size * crop_num
        local_crops = imutils.denormalize_img2(local_crops)
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(-1, 1, crop_size, crop_size)
        local_crops, local_crops_pseudo_label = imutils.augment_data(local_crops, local_crops_pseudo_label)

        # # 降采样减小计算量
        # local_crops = F.interpolate(
        #     local_crops, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        # )

        # Inference
        _, _, _, _, _, _, local_cams = model(local_crops, seg_detach=self.args.seg_detach)
        local_cams = F.interpolate(
            local_cams, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        )

        # Reshape to -> (batch_size, crop_num, C, crop_size // down_scale, crop_size // down_scale)
        local_cams = local_cams.reshape(
            -1, crop_num, 20, crop_size // down_scale, crop_size // down_scale
        )
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(
            -1, crop_num, 1, crop_size, crop_size
        )

        loss_total = 0
        for i in range(self.args.samples_per_gpu):
            local_cams_i = local_cams[i]
            local_crops_pseudo_label_i = F.interpolate(
                local_crops_pseudo_label[i], size=(crop_size // down_scale, crop_size // down_scale), mode='nearest'
            )
            global_segs_i = cam_helper.label_to_aff_mask(local_crops_pseudo_label_i.squeeze(1))
            loss_i = get_masked_ptc_loss(local_cams_i, global_segs_i)

            loss_total += loss_i

        alignment_loss = loss_total / (self.args.samples_per_gpu * crop_num)

        return alignment_loss


class AlignmentLoss_12(nn.Module):
    def __init__(self, args):
        super(AlignmentLoss_12, self).__init__()
        self.args = args

    def forward(self, model, local_crops, local_crops_pseudo_label, down_scale, crop_num, crop_size):
        local_crops = local_crops.reshape(-1, 3, crop_size, crop_size)  # batch_size * crop_num
        n = local_crops.shape[0]
        local_crops = imutils.denormalize_img2(local_crops)
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(-1, 1, crop_size, crop_size)
        local_crops, local_crops_pseudo_label = imutils.augment_data(local_crops, local_crops_pseudo_label)

        # # 降采样减小计算量
        # local_crops = F.interpolate(
        #     local_crops, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        # )

        # Inference
        local_cls, _, _, _, _, local_cams = model(local_crops, seg_detach=self.args.seg_detach)
        local_cams = F.interpolate(
            local_cams, size=(crop_size // down_scale, crop_size // down_scale), mode='bilinear', align_corners=False
        )

        # CLS Loss
        local_crops_pseudo_label_cls = torch.zeros((n, 21), device=local_cls.device).long()
        for i in range(n):
            local_crops_pseudo_label_cls[i][local_crops_pseudo_label[i].squeeze(0).unique().long()[:-1]] = 1

        # Reshape to -> (batch_size, crop_num, C, crop_size // down_scale, crop_size // down_scale)
        local_cams = local_cams.reshape(
            -1, crop_num, 20, crop_size // down_scale, crop_size // down_scale
        )
        local_crops_pseudo_label = local_crops_pseudo_label.reshape(
            -1, crop_num, 1, crop_size, crop_size
        )

        loss_total = 0
        for i in range(self.args.samples_per_gpu):
            local_cams_i = local_cams[i]
            local_crops_pseudo_label_i = F.interpolate(
                local_crops_pseudo_label[i], size=(crop_size // down_scale, crop_size // down_scale), mode='nearest'
            )
            global_segs_i = cam_helper.label_to_aff_mask(local_crops_pseudo_label_i.squeeze(1))
            loss_i = get_masked_ptc_loss(local_cams_i, global_segs_i)

            loss_total += loss_i

        alignment_loss = loss_total / (self.args.samples_per_gpu * crop_num)

        cls_loss = F.multilabel_soft_margin_loss(local_cls, local_crops_pseudo_label_cls[:, 1:])

        return alignment_loss + cls_loss


def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)


class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class RepelLoss(nn.Module):
    def __init__(self, args):
        super(RepelLoss, self).__init__()
        self.args = args
        self.sim_min_loss = SimMinLoss(metric='cos', reduction='mean')

    def forward(self, reps, pseudo_label, cls_label):
        _, _, h_rep, w_rep = reps.shape
        pseudo_label = F.interpolate(pseudo_label.unsqueeze(1).float(), size=(h_rep, w_rep), mode='nearest').squeeze(1)
        reps = reps.permute(0, 2, 3, 1).contiguous()

        repel_loss = torch.zeros(1).to(device=reps.device)
        sum = 0
        for i in range(self.args.samples_per_gpu):
            for c in (torch.nonzero(cls_label[i])):
                target_mask = (pseudo_label[i] == c + 1)
                other_mask = (pseudo_label[i] != c + 1) & (pseudo_label[i] != 255)
                if target_mask.sum() == 0 or other_mask.sum() == 0:  # No representation was selected
                    continue
                reps_i_c = reps[i][target_mask].mean(0)
                reps_i_other = reps[i][other_mask].mean(0)

                repel_loss = repel_loss + self.sim_min_loss(reps_i_c.unsqueeze(0), reps_i_other.unsqueeze(0))
                sum += 1

        repel_loss = repel_loss / sum

        return repel_loss


# Convert CAM label to image-level label
def convert_seg2image_cls(pseudo_label, args, num_classes=20):
    cls_label_cam = torch.zeros(args.samples_per_gpu, num_classes)
    for i in range(args.samples_per_gpu):
        cls_i = torch.unique(pseudo_label[i]).long()
        cls_i = cls_i[(cls_i != 0) & (cls_i != 255)] - 1
        cls_label_cam[i, cls_i] = 1
    cls_label_cam = cls_label_cam.to(pseudo_label.device)

    return cls_label_cam
