import torch
import torch.nn.functional as F
from .imutils import denormalize_img, encode_cmap
from .dcrf import crf_inference_label
import numpy as np


# 通过不同尺度，生成质量更好的 CAM，会上采样至和原始输入图像一样的大小 + Normalization
# Note: 会上采样至
def multi_scale_cam(model, inputs, scales):
    cam_list = []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _ = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _ = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


# 通过不同尺度，生成质量更好的 CAM + 输出aff_mat，会上采样至和原始输入图像一样的大小 + Normalization
def multi_scale_cam_with_aff_mat(model, inputs, scales):
    cam_list, aff_mat = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _aff_mat = model(inputs_cat, cam_only=True)
        aff_mat.append(_aff_mat)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _aff_mat = model(inputs_cat, cam_only=True)
                aff_mat.append(_aff_mat)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    max_aff_mat = aff_mat[np.argmax(scales)]
    return cam, max_aff_mat


# valid_cam - 去除掉无关类别; pseudo_label - 生成伪标签 (仅保留高阈值，不确定区域做 255 处理，低于 low_res 区域做背景处理)
def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, args=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= args.bg_score] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= args.high_thres] = 255
        _pseudo_label[cam_value <= args.low_thres] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * 255

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return valid_cam, pseudo_label


def cam_to_roi_mask(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0
    roi_mask[cam_value >= hig_thre] = 2

    return roi_mask


def crop_from_pseudo_label(images, pseudo_mask=None, crop_num=8, crop_size=96):
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    margin = crop_size // 2

    # Exp
    pseudo_crop_cls = torch.zeros(size=(b, crop_num, 20)).to(images.device)

    for i1 in range(b):
        roi_index = (
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 0) &
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 255)
        ).nonzero()

        if roi_index.shape[0] < crop_num:
            roi_index = (pseudo_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]

            # crop pseudo_labels (exp)
            temp_cls = torch.unique(pseudo_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)])
            filtered_temp_cls = temp_cls[torch.logical_and(temp_cls > 0, temp_cls < 255)] - 1  # filter
            pseudo_crop_cls[i1, i2, filtered_temp_cls] = 1

    return temp_crops, pseudo_crop_cls


def crop_from_pseudo_label_with_cam(images, pseudo_mask=None, cams=None, crop_num=8, crop_size=96):
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    temp_cams = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    margin = crop_size // 2

    # Exp
    pseudo_crop_cls = torch.zeros(size=(b, crop_num, 20)).to(images.device)

    for i1 in range(b):
        roi_index = (
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 0) &
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 255)
        ).nonzero()

        if roi_index.shape[0] < crop_num:
            roi_index = (pseudo_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            temp_cams[i1, i2, ...] = cams[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]

            # crop pseudo_labels (exp)
            temp_cls = torch.unique(pseudo_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)])
            filtered_temp_cls = temp_cls[torch.logical_and(temp_cls > 0, temp_cls < 255)] - 1  # filter
            pseudo_crop_cls[i1, i2, filtered_temp_cls] = 1

    print(temp_cams.grad)
    exit(6)

    return temp_crops, pseudo_crop_cls, temp_cams


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None,
                            down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * cfg.high_thres
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * cfg.low_thres
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * 255
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = 255
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def refine_cams_with_bkg_aff(par, inputs_denorm=None, cams=None, cls_labels=None, aff_mat=None, infer_size=None,
                             attn_mask_infer=None, cfg=None, img_box=None, down_scale=2):
    bkg_cls = torch.ones(size=(cfg.samples_per_gpu, 1))

    b, c, h, w = cams.shape
    cls_label_rep = cls_labels.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cams
    valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)

    # ----------------- cam with aff -----------------
    aff_cam_l = propagte_aff_cam_with_bkg(
        valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels,
        bkg_score=cfg.low_thres
    )
    aff_cam_l = F.interpolate(aff_cam_l, size=(h, w), mode='bilinear', align_corners=False)

    aff_cam_h = propagte_aff_cam_with_bkg(
        valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels,
        bkg_score=cfg.high_thres
    )
    aff_cam_h = F.interpolate(aff_cam_h, size=(h, w), mode='bilinear', align_corners=False)

    # ----------------- refine cam with cls label -----------------
    bkg_cls = bkg_cls.to(cams.device)
    _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_aff_cam_l = refine_cams_with_cls_label(
        par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box
    )
    refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
    refined_aff_cam_h = refine_cams_with_cls_label(
        par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box
    )
    refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

    refined_aff_label = refined_aff_label_h.clone()
    refined_aff_label[refined_aff_label_h == 0] = 255
    refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0
    refined_aff_label = ignore_img_box(refined_aff_label, img_box=img_box, ignore_index=255)

    return refined_aff_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def cams_to_refine_label(cam_label, mask=None, ignore_index=255):
    b, h, w = cam_label.shape

    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h // 16, w // 16],
                                      mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1).contiguous()
    ref_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    # ref_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            ref_label[i, mask == 0] = ignore_index

        ref_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        ref_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index

    return ref_label


def propagte_aff_cam_with_bkg(cams, aff=None, mask=None, cls_labels=None, bkg_score=None):
    b, _, h, w = cams.shape

    bkg = torch.ones(size=(b, 1, h, w)) * bkg_score
    bkg = bkg.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    cams_with_bkg = torch.cat((bkg, cams), dim=1)

    cams_rw = torch.zeros_like(cams_with_bkg)

    ##########

    b, c, h, w = cams_with_bkg.shape
    n_pow = 2.0
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask == 0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1)  ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i, ...])[:, 0]
        _cams = _cams[valid_key, ...]
        _cams = F.softmax(_cams, dim=0)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i, valid_key, :] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw


def ignore_img_box(label, img_box, ignore_index):
    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label


def cams_to_affinity_label(cam_label, mask=None, ignore_index=255):
    b, h, w = cam_label.shape

    cam_label_resized = F.interpolate(cam_label.unsqueeze(1).type(torch.float32), size=[h // 16, w // 16],
                                      mode="nearest")

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1).contiguous()
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    # aff_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            aff_label[i, mask == 0] = ignore_index

        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index

    return aff_label


def refine_cams_with_cls_label(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    # bg_label = torch.ones(size=(b, 1),).to(labels.device)
    cls_label = labels

    for idx, coord in enumerate(img_box):
        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h // 2, w // 2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx, ...])[:, 0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key, ...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0, ...]

    return refined_cams


# def multi_scale_cam_with_dino(model, inputs, scales):
#     cam_list, aff_mat, cam_list_small = [], [], []
#     b, c, h, w = inputs.shape
#     with torch.no_grad():
#         inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
#
#         _cam, _aff_mat = model(inputs_cat, cam_only=True)
#         aff_mat.append(_aff_mat)
#         cam_list_small.append(F.relu(_cam))
#
#         _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
#         _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
#
#         cam_list = [F.relu(_cam)]
#
#         for s in scales:
#             if s != 1.0:
#                 _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
#                 inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
#
#                 _cam, _aff_mat = model(inputs_cat, cam_only=True)
#                 aff_mat.append(_aff_mat)
#                 cam_list_small.append(F.relu(_cam))
#
#                 _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
#                 _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
#
#                 cam_list.append(F.relu(_cam))
#
#         cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
#         cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
#         cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
#
#     max_aff_mat = aff_mat[np.argmax(scales)]
#     return cam, max_aff_mat, cam_list_small


def multi_scale_cam_with_dino(model, inputs, scales):
    cam_list, aff_mat, cam_s = [], [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _aff_mat = model(inputs_cat, cam_only=True)
        aff_mat.append(_aff_mat)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        _cam_tmp = F.relu(_cam)
        _cam_tmp = _cam_tmp + F.adaptive_max_pool2d(-_cam_tmp, (1, 1))
        _cam_tmp /= F.adaptive_max_pool2d(_cam_tmp, (1, 1)) + 1e-5
        cam_s.append(_cam_tmp)

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _aff_mat = model(inputs_cat, cam_only=True)

                aff_mat.append(_aff_mat)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

                _cam_tmp = F.relu(_cam)
                _cam_tmp = _cam_tmp + F.adaptive_max_pool2d(-_cam_tmp, (1, 1))
                _cam_tmp /= F.adaptive_max_pool2d(_cam_tmp, (1, 1)) + 1e-5
                cam_s.append(_cam_tmp)

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    max_aff_mat = aff_mat[np.argmax(scales)]
    return cam, max_aff_mat, cam_s


def label_to_aff_mask(cam_label, ignore_index=255):
    b, h, w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    aff_label[:, range(h * w), range(h * w)] = ignore_index
    return aff_label


def crop_from_pseudo_label_1(images, pseudo_mask=None, crop_num=2, crop_size=128):
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    margin = crop_size // 2

    # Exp
    crop_pseudo_label = torch.zeros(size=(b, crop_num, 1, crop_size, crop_size)).to(images.device)

    for i1 in range(b):
        roi_index = (
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 0) &
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 255)
        ).nonzero()  # 不是背景不是255就是ROI

        if roi_index.shape[0] < crop_num:
            roi_index = (pseudo_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            crop_pseudo_label[i1, i2, ...] = pseudo_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]

    return temp_crops, crop_pseudo_label


def multi_scale_cam_v9(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def refine_cams_with_bkg_v9(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None,
                            down_scale=2, args=None):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * args.high_thres
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * args.low_thres
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * 255
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = 255
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label
