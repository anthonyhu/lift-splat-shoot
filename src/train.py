"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info
from .utils import print_model_spec, set_module_grad

BATCH_SIZE = 2

RAND_FLIP = False  # True for basic
NCAMS = 6  # 5 for basic
PRETRAINED_MODEL_WEIGHTS = './model_weights/model525000.pt'
MODEL_CONFIG = {'receptive_field': 3,
                'n_future': 3,
                'latent_dim': 1,
                'action_as_input': False,
                'temporal_model_name': 'gru',
                'start_out_channels': 80,
                'extra_in_channels': 8,
                'use_pyramid_pooling': False,
                }
SEQUENCE_LENGTH = MODEL_CONFIG['receptive_field'] + MODEL_CONFIG['n_future']
MODEL_NAME = 'temporal'
LEARNING_RATE = 3e-4


def train(version,
            dataroot='/data/cvfs/ah2029/datasets/nuscenes',
            nepochs=10000,
            gpuid=0,

            H=900, W=1600,
            resize_lim=(0.193, 0.225),
            final_dim=(128, 352),
            bot_pct_lim=(0.0, 0.22),
            rot_lim=(-5.4, 5.4),
            rand_flip=RAND_FLIP,
            ncams=NCAMS,
            max_grad_norm=5.0,
            pos_weight=2.13,
            logdir='./runs',

            xbound=[-50.0, 50.0, 0.5],
            ybound=[-50.0, 50.0, 0.5],
            zbound=[-10.0, 10.0, 20.0],
            dbound=[4.0, 45.0, 1.0],

            bsz=BATCH_SIZE,
            nworkers=6,
            lr=LEARNING_RATE,
            weight_decay=1e-7,
            ):
    if torch.cuda.device_count() == 8:
        dataroot = '/mnt/local/datasets/nuscenes'

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }

    if MODEL_NAME == 'temporal':
        parser_name = 'sequentialsegmentationdata'
    else:
        parser_name = 'segmentationdata'
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser_name, sequence_length=SEQUENCE_LENGTH)

    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1, name=MODEL_NAME, model_config=MODEL_CONFIG)

    # Load encoder/decoder weights
    if PRETRAINED_MODEL_WEIGHTS:
        print(f'Loading weights from {PRETRAINED_MODEL_WEIGHTS}')
        # Delete first decoder weight because number of channels might change
        pretrained_model_weights = torch.load(PRETRAINED_MODEL_WEIGHTS)
        del pretrained_model_weights['bevencode.conv1.weight']
        model.load_state_dict(pretrained_model_weights, strict=False)

        print('Freezing image to bev encoder.')
        set_module_grad(model.camencode, requires_grad=True)

    # Print model specs
    print_model_spec(model.camencode, 'Image to BEV encoder')
    print_model_spec(model.temporal_model, 'Temporal model')
    print_model_spec(model.future_prediction, 'Future prediction module')
    print_model_spec(model.bevencode, 'BEV decoder')

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 50 if version == 'mini' else 5000

    model.train()
    counter = 0
    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
            counter = train_step(imgs, rots, trans, intrins, post_rots, post_trans, binimgs, opt, model, device,
                                 loss_fn, max_grad_norm, writer, epoch, valloader, val_step, logdir, counter)


def train_step(imgs, rots, trans, intrins, post_rots, post_trans, binimgs, opt, model, device, loss_fn, max_grad_norm,
               writer, epoch, valloader, val_step, logdir, counter):
    t0 = time()
    opt.zero_grad()
    preds = model(imgs.to(device),
                  rots.to(device),
                  trans.to(device),
                  intrins.to(device),
                  post_rots.to(device),
                  post_trans.to(device),
                  )
    binimgs = binimgs.to(device)

    if MODEL_NAME == 'temporal':
        binimgs = binimgs[:, (model.receptive_field - 1):].contiguous()

        #  Pack sequence dimension
        preds = model.pack_sequence_dim(preds).contiguous()
        binimgs = model.pack_sequence_dim(binimgs).contiguous()

    loss = loss_fn(preds, binimgs)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    opt.step()
    counter += 1
    t1 = time()

    if counter % 10 == 0:
        print(f'Iteration {counter}, loss={loss.item()}, step time ={t1 - t0}')
        writer.add_scalar('train/loss', loss, counter)

    if counter % 50 == 0:
        _, _, iou = get_batch_iou(preds, binimgs)
        writer.add_scalar('train/iou', iou, counter)
        writer.add_scalar('train/epoch', epoch, counter)
        writer.add_scalar('train/step_time', t1 - t0, counter)
        print(f'train iou: {iou}')

    if counter % val_step == 0:
        val_info = get_val_info(model, valloader, loss_fn, device, is_temporal=(MODEL_NAME == 'temporal'))
        print('VAL', val_info)
        writer.add_scalar('val/loss', val_info['loss'], counter)
        writer.add_scalar('val/iou', val_info['iou'], counter)

    if counter % val_step == 0:
        model.eval()
        mname = os.path.join(logdir, "model{}.pt".format(counter))
        print('saving', mname)
        torch.save(model.state_dict(), mname)
        model.train()

    return counter
