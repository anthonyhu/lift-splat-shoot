"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
import matplotlib
matplotlib.use('Agg')

import torch
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx, get_nusc_maps, get_local_map, \
    convert_egopose_to_matrix, mat2pose_vec, compute_future_trajectory
from .utils import convert_figure_numpy
from .constants import DRIVEABLE_AREA_ID, LINE_MARKINGS_ID, N_FUTURE_POINTS

RECEPTIVE_FIELD = 3


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf, sequence_length=0, map_labels=False,
                 dataroot='', output_cost_map=False):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf
        self.sequence_length = sequence_length
        self.map_labels = map_labels
        self.output_cost_map = output_cost_map
        self.dataroot = dataroot
        self.mode = 'train' if self.is_train else 'val'

        if map_labels:
            self.nusc_maps = get_nusc_maps(self.dataroot)
            scene2map = {}
            for rec in self.nusc.scene:
                log = self.nusc.get('log', rec['log_token'])
                scene2map[rec['name']] = log['location']
            self.scene2map = scene2map

            self.driveable_area_id = DRIVEABLE_AREA_ID
            self.line_markings_id = LINE_MARKINGS_ID

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        samp1 = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', samp1['ego_pose_token'])
        yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])  # .inverse
        tran = np.array(pose_record['translation'])[:, None]
        transform = np.vstack((np.hstack((rot.rotation_matrix, tran)), np.array([0, 0, 0, 1])))

        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # go from world to egopose
            pose_record = self.nusc.get('ego_pose', samp['ego_pose_token'])
            rot = Quaternion(pose_record['rotation']).inverse
            tran = -np.array(pose_record['translation'])[:, None]
            transform1 = np.vstack(
                (np.hstack((rot.rotation_matrix, rot.rotation_matrix @ tran)), np.array([0, 0, 0, 1])))

            # # go from egopose to sensor
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = Quaternion(sens['rotation'])#.inverse
            tran = np.array(sens['translation'])[:, None]
            transform2 = np.vstack((np.hstack((rot.rotation_matrix, tran)), np.array([0, 0, 0, 1])))
            transform2 = np.linalg.inv(transform2)

            # transform3 = transform
            # transform3 = transform1 @ transform
            transform3 = transform2 @ transform1 @ transform
            transform3 = np.linalg.inv(transform3)
            rot = torch.Tensor(transform3[:3, :3])
            tran = torch.Tensor(transform3[:3, 3])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        # rot = Quaternion(egopose['rotation']).inverse
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        img1 = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]

            z = box.bottom_corners()[2, 0]
            cv2.fillPoly(img, [pts], 1.0)
            cv2.fillPoly(img1, [pts], z)

        return torch.Tensor(img).long(), torch.Tensor(img1).float()

    def compute_static_label(self, rec, index):
        print(f'Generating static scene for dataset {self.mode} and index={index}')
        dpi = 100
        height, width = (200, 200)
        driveable_area_color = (1.00, 0.50, 0.31)
        line_markings_color = (159. / 255., 0.0, 1.0)

        poly_names = ['road_segment', 'lane']
        line_names = ['road_divider', 'lane_divider']

        dx, bx = self.dx[:2], self.bx[:2]

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        lmap = get_local_map(self.nusc_maps[map_name], center,
                             50.0, poly_names, line_names)


        label = np.zeros((height, width), dtype=np.int64)

        # Driveable area
        fig = plt.figure(dpi=dpi)
        ax = fig.gca()
        ax.set_axis_off()

        for name in poly_names:
            for la in lmap[name]:
                pts = (la - bx) / dx
                ax.fill(width - pts[:, 1], height - pts[:, 0], c=driveable_area_color, linewidth=.01)

        ax.set_xlim((width, 0))
        ax.set_ylim((0, height))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.set_figwidth(height / dpi)
        fig.set_figheight(width / dpi)

        plt.draw()
        plt.close('all')

        fig_np = convert_figure_numpy(fig)

        label[fig_np.sum(axis=2) < 255 * 3] = self.driveable_area_id

        # Line markings
        fig = plt.figure(dpi=dpi)
        ax = fig.gca()
        ax.set_axis_off()

        for name in line_names:
            for la in lmap[name]:
                pts = (la - bx) / dx
                ax.plot(width - pts[:, 1], height - pts[:, 0], c=line_markings_color, linewidth=.01)

        ax.set_xlim((width, 0))
        ax.set_ylim((0, height))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.set_figwidth(height / dpi)
        fig.set_figheight(width / dpi)

        plt.draw()
        plt.close('all')

        fig_np = convert_figure_numpy(fig)

        label[fig_np.sum(axis=2) < 255 * 3] = self.line_markings_id

        label = torch.Tensor(label).long()
        return label

    def get_dynamic_label(self, rec):
        return self.get_binimg(rec)

    def get_static_label(self, rec, index):
        if not self.map_labels:
            return None

        # Load saved labels
        label_path = os.path.join(self.dataroot, 'static_label', self.mode, f'static_label_{index:08d}.png')

        if os.path.isfile(label_path):
            static_label = np.asarray(Image.open(label_path)).astype(np.int64)
            static_label = torch.Tensor(static_label).long()
        else:
            static_label = self.compute_static_label(rec, index)

        return static_label

    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = \
                    self.nusc.get('ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token'])
                egopose_t1 = \
                    self.nusc.get('ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token'])

                egopose_t0 = convert_egopose_to_matrix(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix(egopose_t1)

                future_egomotion = np.linalg.inv(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion

    def get_cost_map_trajectory(self, index):
        trajectory, valid_sequence = compute_future_trajectory(self, index, n_future_points=N_FUTURE_POINTS)
        if valid_sequence:
            trajectory = torch.Tensor(trajectory).float()
        else:
            trajectory = torch.zeros(10, 2)
        return trajectory

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg, z_position = self.get_dynamic_label(rec)
        if self.map_labels:
            static_label = self.get_static_label(rec, index)
        else:
            static_label = []
        future_egomotion = self.get_future_egomotion(rec, index)
        if self.output_cost_map:
            future_trajectory = self.get_cost_map_trajectory(index)
        else:
            future_trajectory = []

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, static_label, future_egomotion, \
               future_trajectory, z_position


class SequentialSegmentationData(SegmentationData):
    def __getitem__(self, index):
        list_imgs, list_rots, list_trans, list_intrins = [], [], [], []
        list_post_rots, list_post_trans, list_binimg, list_static_label, list_future_egomotion = [], [], [], [], []
        list_z_position, list_recs = [], []
        cams = self.choose_cams()

        previous_rec = None
        index_t = None
        previous_index_t = None
        for t in range(self.sequence_length):
            if index + t >= len(self):
                rec = previous_rec
                index_t = previous_index_t
            else:
                rec = self.ixes[index + t]
                index_t = index + t

                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    # Repeat image
                    rec = previous_rec
                    index_t = previous_index_t

            imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
            binimg, z_position = self.get_dynamic_label(rec)
            if self.map_labels:
                static_label = self.get_static_label(rec, index_t)
            future_egomotion = self.get_future_egomotion(rec, index_t)

            list_imgs.append(imgs)
            list_rots.append(rots)
            list_trans.append(trans)
            list_intrins.append(intrins)
            list_post_rots.append(post_rots)
            list_post_trans.append(post_trans)
            list_binimg.append(binimg)
            list_z_position.append(z_position)
            list_recs.append(rec['token'])
            if self.map_labels:
                list_static_label.append(static_label)
            list_future_egomotion.append(future_egomotion)

            previous_rec = rec
            previous_index_t = index_t

        if self.output_cost_map:
            time_offset = RECEPTIVE_FIELD - 1
            future_trajectory = self.get_cost_map_trajectory(index + time_offset)
            future_trajectory.unsqueeze(1)
        else:
            future_trajectory = []

        list_imgs, list_rots, list_trans, list_intrins = torch.stack(list_imgs), torch.stack(list_rots), \
                                                         torch.stack(list_trans), torch.stack(list_intrins)

        list_post_rots, list_post_trans, list_binimg = torch.stack(list_post_rots), torch.stack(list_post_trans), \
                                                       torch.stack(list_binimg)
        list_z_position = torch.stack(list_z_position)
        if self.map_labels:
            list_static_label = torch.stack(list_static_label)
        list_future_egomotion = torch.stack(list_future_egomotion)

        return (list_imgs, list_rots, list_trans, list_intrins, list_post_rots, list_post_trans, list_binimg,
                list_static_label, list_future_egomotion, future_trajectory, list_z_position, list_recs)


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name, sequence_length=0, map_labels=False, output_cost_map=False):
    dataroot = os.path.join(dataroot, version)

    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
        'sequentialsegmentationdata': SequentialSegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf, sequence_length=sequence_length, map_labels=map_labels,
                       dataroot=dataroot, output_cost_map=output_cost_map)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf, sequence_length=sequence_length, map_labels=map_labels,
                     dataroot=dataroot, output_cost_map=output_cost_map)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
