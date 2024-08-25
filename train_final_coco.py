'''
FINAL TRAINING SCRIPT of MS COCO
'''

import numpy as np
import torch
import random
import cv2
import os
import datetime
import torch.nn.functional as F
import argparse
import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# Modules of ECA
# ----------------------------------------------------------------
from test_final_eval import final_eval
from modules.PAR import PAR
from modules.model_attn_aff_v8 import WeTr
from datasets import coco
from utils import train_helper, cam_helper, AverageMeter, imutils, pyutils

from dino import vision_transformer_v1 as vits
from modules.label_from_dino import get_pseudo_labels_rv_cam_10_4

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
local_rank = int(os.environ['LOCAL_RANK'])


def train(args):
    # Init
    # ------------------------------------------------------------------------------------------------
    torch.cuda.set_device(local_rank)
    device = torch.device(local_rank)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    if local_rank == 0:
        tblogger = SummaryWriter(comment=args.comment, log_dir=args.tb_log_dir)  # debug

    train_dataset = coco.CocoClsDataset(
        root_dir=args.root_dir, name_list_dir=args.name_list_dir, split=args.split,
        stage='train', aug=True, resize_range=[512, 2048], rescale_range=[0.5, 2.0],
        crop_size=args.crop_size, img_fliplr=True,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=args.samples_per_gpu, shuffle=False, num_workers=args.num_workers, pin_memory=False,
        drop_last=True, sampler=train_sampler, prefetch_factor=4
    )
    train_sampler.set_epoch(np.random.randint(args.max_iter))
    train_loader_iter = iter(train_loader)

    val_dataset = coco.CocoSegDataset(
        root_dir=args.root_dir, name_list_dir=args.name_list_dir, split='val_part',
        stage='val', aug=False,
    )  # only use 5k images for fast validation (val_part)

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False, drop_last=False
    )

    # Models
    # ------------------------------------------------------------------------------------------------
    # SegFormer
    model = WeTr(
        backbone='mit_b1', stride=[4, 2, 2, 1], num_classes=81,
        pretrained_dir='/media/store/wyc/exps/T_Wseg/pretrained/mit_b1.pth',
        embedding_dim=args.embedding_dim, pooling=args.pooling
    ).to(device)
    param_groups = model.get_param_groups()

    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # DINO
    model_dino = vits.__dict__['vit_small'](patch_size=args.patch_size, num_classes=0)
    for p in model_dino.parameters():
        p.requires_grad = False
    model_dino.eval()
    model_dino.to(device)
    state_dict = torch.load('./checkpoint/dino_deitsmall{}_pretrain.pth'.format(args.patch_size), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    model_dino.load_state_dict(state_dict, strict=False)

    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).to(device)

    optimizer = train_helper.get_optimizer(
        param_groups, args.lr, args.weight_decay,
        args.warmup_iter, args.max_iter, args.warmup_ratio, args.power
    )

    avg_meter = AverageMeter.AverageMeter('loss', 'cls_loss', 'seg_loss_d', 'aff_loss', 'seg_loss_c', 'ali_loss')

    mask_size = int(args.crop_size // 16)
    infer_size = int((args.crop_size * max([1, 0.5, 1.5])) // 16)
    attn_mask = train_helper.get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    attn_mask_infer = train_helper.get_mask_by_radius(h=infer_size, w=infer_size, radius=args.radius)

    memobank = torch.randn(80, 512).to(device)

    # Train
    # ------------------------------------------------------------------------------------------------
    for n_iter in range(args.max_iter):
        if local_rank == 0:
            pyutils.print_progress(n_iter, args.max_iter)

        if n_iter == args.cam_iters:
            avg_meter.pop('seg_loss_d')
            avg_meter.pop('aff_loss')
            avg_meter.pop('seg_loss_c')
            avg_meter.pop('ali_loss')

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(args.max_iter))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)

        # n_iter = 6666  # debug only

        if n_iter <= args.cam_iters:
            # Inference
            # ------------------------------------------------------------------------------------------------
            cls = model(inputs, cls_only=True)
            cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)
            seg_loss_d = torch.ones(1)
            aff_loss = torch.ones(1)
            seg_loss_c = torch.ones(1)
            ali_loss = torch.ones(1)

            loss = 1.0 * cls_loss

        else:
            # Inference
            # ------------------------------------------------------------------------------------------------
            cls, segs, attns, attn_pred, reps, _ = model(inputs, seg_detach=args.seg_detach)
            cams, aff_mat, _ = cam_helper.multi_scale_cam_with_dino(
                model, inputs=inputs, scales=[1.0, 0.5, 1.5]
            )

            cam_pseudo_label = cam_helper.refine_cams_with_bkg_v2(
                par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=args, img_box=img_box
            )

            cls_labels_cam = train_helper.convert_seg2image_cls(cam_pseudo_label, args, num_classes=80)

            dino_pseudo_label = get_pseudo_labels_rv_cam_10_4(
                model_dino, inputs, cls_labels_cam, cams, reps, par, [1.0, 0.5], img_box, memobank, args, num_class=81
            )

            # Classification loss
            # ------------------------------------------------------------------------------------------------
            cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

            # Affinity Loss
            # ------------------------------------------------------------------------------------------------
            aff_label_d = cam_helper.cams_to_affinity_label(dino_pseudo_label, mask=attn_mask, ignore_index=255)
            aff_loss_1, _, _ = train_helper.get_aff_loss(attn_pred, aff_label_d)
            aff_label_c = cam_helper.cams_to_affinity_label(cam_pseudo_label, mask=attn_mask, ignore_index=255)
            aff_loss_2, _, _ = train_helper.get_aff_loss(attn_pred, aff_label_c)
            aff_loss = 0.5 * (aff_loss_1 + aff_loss_2)

            # Segmentation Loss (DINO)
            # ------------------------------------------------------------------------------------------------
            segs = F.interpolate(segs, size=dino_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            seg_loss_d = train_helper.get_seg_loss_v9(segs, dino_pseudo_label.type(torch.long), ignore_index=255)

            # Segmentation Loss (CAM)
            # ------------------------------------------------------------------------------------------------
            segs = F.interpolate(segs, size=cam_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            # if n_iter <= 15000:
            #     seg_loss_c = train_helper.get_seg_loss_v9(segs, cam_pseudo_label.type(torch.long), ignore_index=255)
            # else:
            #     cam_pseudo_label_aff = cam_helper.refine_cams_with_bkg_aff(
            #         par, inputs_denorm, cams=cams, cls_labels=cls_labels, aff_mat=aff_mat, infer_size=infer_size,
            #         attn_mask_infer=attn_mask_infer, cfg=args, img_box=img_box
            #     )
            #     seg_loss_c = train_helper.get_seg_loss_v9(segs, cam_pseudo_label_aff.type(torch.long), ignore_index=255)
            seg_loss_c = train_helper.get_seg_loss_v9(segs, cam_pseudo_label.type(torch.long), ignore_index=255)

            # CAM alignment
            # ------------------------------------------------------------------------------------------------
            local_crops, local_crops_pseudo_label = cam_helper.crop_from_pseudo_label_1(
                images=inputs, pseudo_mask=cam_pseudo_label.type(torch.long), crop_num=args.patch_num,
                crop_size=args.crop_patch_size
            )
            alignment_loss = train_helper.AlignmentLoss(args, num_classes=80)
            ali_loss = alignment_loss(
                model, local_crops, local_crops_pseudo_label, down_scale=4,
                crop_num=args.patch_num, crop_size=args.crop_patch_size
            )

            # Loss calculation
            # ------------------------------------------------------------------------------------------------
            if n_iter <= 10000:
                loss = 1.0 * cls_loss + 0.1 * seg_loss_d + 0.0 * seg_loss_c + \
                       (args.w_aff * aff_loss) + (0.0 * ali_loss)
            else:
                loss = 1.0 * cls_loss + 0.1 * seg_loss_d + 0.1 * seg_loss_c + \
                       (args.w_aff * aff_loss) + (args.w_ali * ali_loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meter.add({
            'loss': loss.item(),
            'cls_loss': cls_loss.item(),
            'seg_loss_d': seg_loss_d.item(),
            'aff_loss': aff_loss.item(),
            'seg_loss_c': seg_loss_c.item(),
            'ali_loss': ali_loss.item(),
        })

        if local_rank == 0:
            # Validation
            # ------------------------------------------------------------------------------------------------
            if (n_iter + 1) % 5000 == 0:
                # torch.save(model.state_dict(), 'v1_iter{}.pth')
                cls_score, seg_score, cam_score, aff_score = train_helper.validate_coco(
                    model=model, data_loader=val_loader, cfg=args
                )

                logging.info(
                    'Evaluation of {} iter: \n\ncls_score: {:.2f} \n\nseg_score: {} \n\ncam_score:{} \n\naff_cam:{}\n'
                    .format(
                        n_iter + 1, cls_score, seg_score, cam_score, aff_score
                    )
                )

                score_dict = {
                    'Seg_score': seg_score['miou'],
                    'Cam_score': cam_score['miou'],
                    'Aff_score': aff_score['miou'],
                }
                tblogger.add_scalars('Validation', score_dict, n_iter + 1)  # debug only

            # Log
            # ------------------------------------------------------------------------------------------------
            if (n_iter + 1) % 200 == 0:
                delta, eta = train_helper.cal_eta(time0, n_iter + 1, args.max_iter)
                logging.info(
                    "iteration: {} [{} : {}]".format(n_iter + 1, delta, eta),
                )
                logging.info(
                    'Total Loss: {:.4f} | cls_loss: {:.4f} | '
                    'aff_loss: {:.4f} | seg_loss_d: {:.4f} | seg_loss_c: {:.4f} | ali_loss: {:.4f}'.format(
                        *avg_meter.get('loss', 'cls_loss', 'aff_loss', 'seg_loss_d', 'seg_loss_c', 'ali_loss')
                    )
                )

            if n_iter % 50 == 0:
                loss_dict = {
                    'total_loss': loss.item(),
                    'cls_loss': cls_loss.item(),
                    'seg_loss_d': seg_loss_d.item(),
                    'aff_loss': aff_loss.item(),
                    'seg_loss_c': seg_loss_c.item(),
                    'ali_loss': ali_loss.item(),
                }
                tblogger.add_scalars('Loss', loss_dict, n_iter + 1)  # debug only
                pass

    logging.info('FINISHED...')

    if local_rank == 0:
        ckpt_name = os.path.join(args.ckpt_dir, '{}.pth'.format(args.comment))
        torch.save(model.state_dict(), ckpt_name)

        logging.info('\n----------------------------- FINAL RESULTS: -----------------------------')
        seg_score, msc_seg_score = final_eval(model, args)
        logging.info("\nsegs score:")
        logging.info(seg_score)
        logging.info("\nmsc segs score:")
        logging.info(msc_seg_score)
        tblogger.add_text('\nfinal_seg_score', str(seg_score))  # debug
        tblogger.add_text('\nfinal_msc_seg_score', str(msc_seg_score))  # debug


# Config
# ------------------------------------------------------------------------------------------------
def get_config():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument("--work_dir", default='./work_dir_coco/', type=str, help="work_dir")
    parser.add_argument("--ckpt_dir", default='checkpoint', type=str, help="ckpt_dir")
    parser.add_argument("--tb_log_dir", default='log', type=str, help="ckpt_dir")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
    parser.add_argument('--backend', default='nccl')
    parser.add_argument("--comment", default='coco_final', type=str, help="comment")

    # Backbone
    parser.add_argument("--embedding_dim", default=256, type=int, help="Backbone")
    parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")

    # Dataset
    parser.add_argument("--root_dir", default="/media/store/wyc/Dataset/MSCOCO", type=str)
    parser.add_argument("--name_list_dir", default="datasets/coco", type=str)

    # Preprocessing
    parser.add_argument("--radius", default=8, type=int, help="radius")
    parser.add_argument("--crop_size", default=448, type=int, help="crop_size")
    parser.add_argument("--num_workers", default=10, type=int)

    # Train
    parser.add_argument("--split", default="train", type=str)
    parser.add_argument("--samples_per_gpu", default=4, type=int)
    parser.add_argument("--lr", default=6e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_iter", default=1500, type=int)
    parser.add_argument("--warmup_ratio", default=1e-6, type=float)
    parser.add_argument("--power", default=1.0, type=float)
    parser.add_argument("--seg_detach", action="store_true", help="detach seg")

    # Framework
    parser.add_argument("--bg_score", default=0.45, type=float)
    parser.add_argument("--high_thres", default=0.55, type=float)
    parser.add_argument("--low_thres", default=0.35, type=float)
    parser.add_argument("--max_iter", default=80000, type=int)
    parser.add_argument("--cam_iters", default=5000, type=int)

    # Loss weights
    parser.add_argument("--w_aff", default=0.1, type=float)
    parser.add_argument("--w_ali", default=0.1, type=float)

    # EXP
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--topk", default=0.3, type=float)
    parser.add_argument("--momentum", default=0.99, type=float)  # 0.9
    parser.add_argument("--thres_cls_token", default=-0.5, type=float)
    parser.add_argument("--thres_patch_token", default=0.15, type=float)
    parser.add_argument("--thres_bg_token", default=0.15, type=float)
    parser.add_argument("--thres_bg_aff_l", default=0.1, type=float)
    parser.add_argument("--thres_bg_aff_h", default=0.6, type=float)

    parser.add_argument("--crop_patch_size", default=256, type=int)
    parser.add_argument("--patch_num", default=2, type=int)  # 2

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_config()

    args.ckpt_dir = os.path.join(args.work_dir, args.ckpt_dir)
    args.tb_log_dir = os.path.join(args.work_dir, args.tb_log_dir)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)

    dist.init_process_group(backend=args.backend, )

    if local_rank == 0:
        print('-----------------------------------------------------------------------------------')
        timestamp = "{0:%Y-%m-%d %H:%M}".format(datetime.datetime.now())
        pyutils.setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info(timestamp)
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s" % (torch.cuda.get_device_name(0)))
        logging.info("Total gpus: %d, samples per gpu: %d" % (dist.get_world_size(), args.samples_per_gpu))
        logging.info('\nargs: %s \n' % args)

    train(args)
