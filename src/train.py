"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import datetime
import socket
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .losses import probabilistic_kl_loss, cost_map_loss, CrossEntropyLoss
from .tools import get_batch_iou, compute_miou, get_val_info, pose_vec2mat, compute_egomotion_error, \
    compute_egomotion_error_plane, load_template_trajectories
from .utils import print_model_spec, set_module_grad

BATCH_SIZE = 1
TAG = 'top_k_gru_kl=0.5_warping_but_no_loss'
OUTPUT_PATH = './runs/debug'

OUTPUT_COST_MAP = False
PREDICT_FUTURE_EGOMOTION = False
WARMSTART_STEPS = 0
VAL_STEPS = 5000
DIRECT_TRAJECTORY_PREDICTION = False
PRETRAINED_MODEL_WEIGHTS = './model_weights/model525000.pt'

USE_TOP_K = True
TOP_K_RATIO = 0.5

TEMPORAL_MODEL_NAME = 'gru'
RECEPTIVE_FIELD = 3
N_FUTURE = 3

LOSS_WEIGHTS = {'dynamic_agents': 1.0,
                'static_agents': 0.5,
                'future_egomotion': 0.0,
                'kl': 0.5,
                'autoregressive': 0.0,
                'cost_map': 1.0,
                }

if 'direwolf' in socket.gethostname():
    RECEPTIVE_FIELD = 2
    N_FUTURE = 2

MODEL_NAME = 'temporal'
PROBABILISTIC = True
THREE_DOF_EGOMOTION = True
DISABLE_BEV_PREDICTION = False
AUTOREGRESSIVE_L2_LOSS = False
AUTOREGRESSIVE_FUTURE_PREDICTION = False
MODEL_CONFIG = {'receptive_field': RECEPTIVE_FIELD,
                'n_future': N_FUTURE,
                'latent_dim': 16,
                'probabilistic': PROBABILISTIC,
                'autoregressive_future_prediction': AUTOREGRESSIVE_FUTURE_PREDICTION,
                'autoregressive_l2_loss': AUTOREGRESSIVE_L2_LOSS,
                'direct_trajectory_prediction': DIRECT_TRAJECTORY_PREDICTION,
                'predict_future_egomotion': PREDICT_FUTURE_EGOMOTION,
                'output_cost_map': OUTPUT_COST_MAP,
                'three_dof_egomotion': THREE_DOF_EGOMOTION,
                'temporal_model_name': TEMPORAL_MODEL_NAME,
                'disable_bev_prediction': DISABLE_BEV_PREDICTION,
                'start_out_channels': 80,
                'extra_in_channels': 8,
                'use_pyramid_pooling': False,
                }
SEQUENCE_LENGTH = MODEL_CONFIG['receptive_field'] + MODEL_CONFIG['n_future']

LEARNING_RATE = 3e-4
N_CLASSES = 2
MAP_LABELS = False
RAND_FLIP = False  # True for basic
NCAMS = 6  # 5 for basic
DATAROOT = '/data/cvfs/ah2029/datasets/nuscenes'
CROSS_ENTROPY_WEIGHTS = [1.0, 2.13]
if MAP_LABELS:
    N_CLASSES = 4
    CROSS_ENTROPY_WEIGHTS = [1.0, 3.0, 1.0, 2.0]


def train(version,
          dataroot=DATAROOT,
          nepochs=10000,

          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=RAND_FLIP,
          ncams=NCAMS,
          max_grad_norm=5.0,
          weight=CROSS_ENTROPY_WEIGHTS,
          tag=TAG,
          output_path=OUTPUT_PATH,
          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=BATCH_SIZE,
          nworkers=5,
          lr=LEARNING_RATE,
          weight_decay=1e-7,
          ):
    logdir = create_session_name(output_path, tag)

    print('Model config:')
    print(MODEL_CONFIG)
    print(LOSS_WEIGHTS)
    print(f'Number of classes: {N_CLASSES}')
    print(f'Warm start for {WARMSTART_STEPS} steps')
    print(f'Session: {logdir}')

    if 'vm' in socket.gethostname():
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
                                          parser_name=parser_name, sequence_length=SEQUENCE_LENGTH,
                                          map_labels=MAP_LABELS, output_cost_map=OUTPUT_COST_MAP)

    device = torch.device('cuda:0')

    model = compile_model(grid_conf, data_aug_conf, outC=N_CLASSES, name=MODEL_NAME, model_config=MODEL_CONFIG)

    model.to(device)

    # Load encoder/decoder weights
    if PRETRAINED_MODEL_WEIGHTS:
        print(f'Loading weights from {PRETRAINED_MODEL_WEIGHTS}')
        # Delete first decoder weight because number of channels might change
        pretrained_model_weights = torch.load(PRETRAINED_MODEL_WEIGHTS)
        del pretrained_model_weights['bevencode.conv1.weight']
        del pretrained_model_weights['bevencode.up2.4.weight']
        del pretrained_model_weights['bevencode.up2.4.bias']
        model.load_state_dict(pretrained_model_weights, strict=False)

        if MODEL_NAME == 'temporal':
            pass
            #print('Freezing image to bev encoder.')
            #set_module_grad(model.camencode, requires_grad=False)

    #  Print model specs
    print_model_spec(model.camencode, 'Image to BEV encoder')
    if MODEL_NAME == 'temporal':
        print_model_spec(model.temporal_model, 'Temporal model')
        print_model_spec(model.future_prediction, 'Future prediction module')

    print_model_spec(model.bevencode, 'BEV decoder')
    if PREDICT_FUTURE_EGOMOTION:
        print_model_spec(model.pose_net, 'Pose net')
    print_model_spec(model, 'Total')

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses_fn = {}
    losses_fn['dynamic_agents'] = CrossEntropyLoss(
        weight=torch.Tensor(weight).to(device), use_top_k=USE_TOP_K, top_k_ratio=TOP_K_RATIO
    )

    if PREDICT_FUTURE_EGOMOTION:
        losses_fn['future_egomotion'] = torch.nn.MSELoss()

    if PROBABILISTIC:
        losses_fn['kl'] = probabilistic_kl_loss

    if AUTOREGRESSIVE_L2_LOSS:
        losses_fn['autoregressive'] = torch.nn.MSELoss()

    if model.output_cost_map:
        losses_fn['cost_map'] = cost_map_loss

    writer = SummaryWriter(logdir=logdir)
    val_step = 10 if version == 'mini' else VAL_STEPS
    train_eval_step = 10 if version == 'mini' else 100

    model.train()
    counter = 0

    if model.output_cost_map:
        templates = load_template_trajectories(device)

    for epoch in range(nepochs):
        np.random.seed()
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, static_labels, future_egomotions, future_trajectory
                     ) in \
                enumerate(
                trainloader):

            t0 = time()
            opt.zero_grad()

            finished_warmstart = counter > WARMSTART_STEPS
            out = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          future_egomotions.to(device),
                          inference=finished_warmstart,
                          )

            if not DISABLE_BEV_PREDICTION:
                preds = out['bev']
            binimgs = binimgs.to(device)

            if MODEL_NAME == 'temporal':
                binimgs = binimgs[:, (model.receptive_field - 1):].contiguous()

                #  Pack sequence dimension
                if not DISABLE_BEV_PREDICTION:
                    preds = model.pack_sequence_dim(preds).contiguous()
                binimgs = model.pack_sequence_dim(binimgs).contiguous()

                if PREDICT_FUTURE_EGOMOTION:
                    future_egomotions = future_egomotions.to(device)
                    future_egomotions = future_egomotions[:, (model.receptive_field - 1):].contiguous()

            losses = {}
            if not DISABLE_BEV_PREDICTION:
                losses['dynamic_agents'] = losses_fn['dynamic_agents'](preds, binimgs)

            if PREDICT_FUTURE_EGOMOTION:
                if model.three_dof_egomotion:
                    # x-y translation and z-axis rotation
                    pose_slice = [0, 1, 5]
                else:
                    pose_slice = list(range(6))
                losses['future_egomotion'] = torch.zeros(1, dtype=torch.float32).to(device)
                #losses_fn['future_egomotion'](out['future_egomotions'][:, :, pose_slice],
                                              #                             future_egomotions[:, :, pose_slice])

            if PROBABILISTIC:
                losses['kl'] = losses_fn['kl'](out)

            if AUTOREGRESSIVE_L2_LOSS:
                losses['autoregressive'] = losses_fn['autoregressive'](out['z'][:, model.receptive_field:],
                                                                       out['z_future_pred'])

            if model.output_cost_map:
                losses['cost_map'] = losses_fn['cost_map'](out, future_trajectory, templates)

            # Calculate total loss
            loss = torch.zeros(1, dtype=torch.float32).to(device)

            for key, value in losses.items():
                loss += LOSS_WEIGHTS[key] * value

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(f'Iteration {counter}, total loss={loss.item()}, step time ={t1 - t0}')
                writer.add_scalar('train/total_loss', loss.item(), counter)
                for key, value in losses.items():
                    print(f'train/{key}_loss={LOSS_WEIGHTS[key] * value.item()}')
                    writer.add_scalar(f'train/{key}_loss', LOSS_WEIGHTS[key] * value.item(), counter)

            if counter % train_eval_step == 0:
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)
                #_, _, iou = get_batch_iou(preds, binimgs.unsqueeze(1))
                if not DISABLE_BEV_PREDICTION:
                    vehicles_iou = compute_miou(
                        torch.argmax(preds, dim=1).float().detach().cpu().numpy(),
                        binimgs.cpu().numpy(),
                        n_classes=N_CLASSES,
                    )
                    writer.add_scalar('train/vehicles_iou', vehicles_iou['vehicles'], counter)
                    print(f'train vehicle iou: {vehicles_iou}')

                if PREDICT_FUTURE_EGOMOTION:
                    if model.three_dof_egomotion:
                        positional_error, angular_error = compute_egomotion_error_plane(
                            out['future_egomotions'].detach().cpu().numpy(), future_egomotions.cpu().numpy()
                        )
                    else:
                        # Convert predicted 6 DoF egomotion to pose matrix
                        predicted_pose_matrices = pose_vec2mat(out['future_egomotions'])
                        gt_pose_matrices = pose_vec2mat(future_egomotions)

                        positional_error, angular_error = compute_egomotion_error(
                            predicted_pose_matrices.detach().cpu().numpy(), gt_pose_matrices.cpu().numpy()
                        )
                    writer.add_scalar('train/positional_error', positional_error, counter)
                    writer.add_scalar('train/angular_error', angular_error, counter)
                    print(f'train positional_error (in m): {positional_error}')
                    print(f'train angular_error (in degrees): {angular_error}')

            if counter % val_step == 0:
                val_info = get_val_info(model, valloader, losses_fn, device, is_temporal=(MODEL_NAME == 'temporal'),
                                        n_classes=N_CLASSES, loss_weights=LOSS_WEIGHTS)
                print('VAL', val_info)
                writer.add_scalar('val/loss', val_info['loss'], counter)
                writer.add_scalar('val/vehicles_iou', val_info['vehicles_iou'], counter)
                writer.add_scalar('val/positional_error', val_info['positional_error'], counter)
                writer.add_scalar('val/angular_error', val_info['angular_error'], counter)

            if counter % val_step == 0:
                model.eval()
                mname = os.path.join(logdir, "model{}.pt".format(counter))
                print('saving', mname)
                torch.save(model.state_dict(), mname)
                model.train()


def create_session_name(output_path, tag):
    now = datetime.datetime.now()
    session_name = 'session_{}_{:04d}_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}_{}'.format(
        socket.gethostname(), now.year, now.month, now.day, now.hour, now.minute, now.second, tag)
    session_name = os.path.join(output_path, session_name)
    os.makedirs(session_name)
    return session_name
