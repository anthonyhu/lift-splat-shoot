"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from src.layers.convolutions import ResBlock
from src.layers.temporal import SpatialGRU, Bottleneck3D, TemporalBlock
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    @staticmethod
    def pack_sequence_dim(x):
        b, s = x.shape[:2]
        return x.view(b * s, *x.shape[2:])

    @staticmethod
    def unpack_sequence_dim(x, b, s):
        return x.view(b, s, *x.shape[1:])

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, future_egomotions):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


class TemporalModel(nn.Module):
    def __init__(self, in_channels, receptive_field, name, start_out_channels=80,
                 extra_in_channels=8, use_pyramid_pooling=True,
                 input_shape=(10, 24), norm='bn', activation='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.name = name
        self.input_shape = input_shape
        self.norm = norm
        self.activation = activation
        self._receptive_field = receptive_field
        self.start_out_channels = start_out_channels
        self.extra_in_channels = extra_in_channels
        self.use_pyramid_pooling = use_pyramid_pooling

        self.n_temporal_layers = receptive_field - 1
        self.n_spatial_layers_between_temporal_layers = 3

        self.model = self.create_model()

    def create_model(self):
        if self.name == 'gru':
#            return SpatialGRU(self.in_channels, self.in_channels, norm=self.norm, activation=self.activation)
            return nn.Sequential(SpatialGRU(self.in_channels, self.in_channels, norm=self.norm,
                                            activation=self.activation),
                                 SpatialGRU(self.in_channels, self.in_channels, norm=self.norm,
                                            activation=self.activation),
                                 SpatialGRU(self.in_channels, self.in_channels, norm=self.norm,
                                            activation=self.activation),
                                 )

        h, w = self.input_shape
        modules = []

        block_in_channels = self.in_channels
        block_out_channels = self.start_out_channels

        for _ in range(self.n_temporal_layers):
            if self.use_pyramid_pooling:
                use_pyramid_pooling = True
                pool_sizes = [(2, h, w), (2, h // 2, w // 2), (2, h // 4, w // 4)]
            else:
                use_pyramid_pooling = False
                pool_sizes = None
            temporal = TemporalBlock(block_in_channels, block_out_channels, use_pyramid_pooling=use_pyramid_pooling,
                                     pool_sizes=pool_sizes)
            spatial = [Bottleneck3D(block_out_channels, block_out_channels, kernel_size=(1, 3, 3))
                       for _ in range(self.n_spatial_layers_between_temporal_layers)]
            temporal_spatial_layers = nn.Sequential(temporal, *spatial)
            modules.extend(temporal_spatial_layers)

            block_in_channels = block_out_channels
            block_out_channels += self.extra_in_channels
        return nn.Sequential(*modules)

    def forward(self, x):
        if self.name == 'gru':
            return self.model(x)
        else:
            # Reshape input tensor to (batch, C, time, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            z = self.model(x)
            return z.permute(0, 2, 1, 3, 4).contiguous()

    @property
    def receptive_field(self):
        return self._receptive_field


class FuturePrediction(torch.nn.Module):
    def __init__(self, in_channels, latent_dim, predict_future_egomotion=False, n_gru_blocks=3, n_res_layers=3):
        super().__init__()
        self.predict_future_egomotion = predict_future_egomotion
        self.n_gru_blocks = n_gru_blocks

        # Convolutional recurrent model with z_t as an initial hidden state and inputs the sample
        # from the probabilistic model. The architecture of the model is:
        # [Spatial GRU - [Bottleneck] x n_res_layers] x n_gru_blocks
        self.spatial_grus = []
        self.res_blocks = []

        for i in range(self.n_gru_blocks):
            gru_in_channels = latent_dim if i == 0 else in_channels
            self.spatial_grus.append(SpatialGRU(gru_in_channels, in_channels))
            self.res_blocks.append(torch.nn.Sequential(*[ResBlock(in_channels)
                                                         for _ in range(n_res_layers)]))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)

    def forward(self, x, hidden_state, future_egomotions):
        # pylint: disable=arguments-differ
        # x has shape (b, n_future, c, h, w), hidden_state (b, c, h, w)
        for i in range(self.n_gru_blocks):

            if self.predict_future_egomotion and i == 0:
                # Warp features with respect to future ego-motion
                flow = future_egomotions
            else:
                flow = None

            x = self.spatial_grus[i](x, hidden_state, flow)
            b, n_future, c, h, w = x.shape

            x = self.res_blocks[i](x.view(b * n_future, c, h, w))
            x = x.view(b, n_future, c, h, w)
        return x


class TemporalLiftSplatShoot(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC, model_config):
        super().__init__(grid_conf, data_aug_conf, outC)

        self.receptive_field = model_config['receptive_field']  # 3
        self.n_future = model_config['n_future']  # 5
        self.latent_dim = model_config['latent_dim']  # 16
        self.predict_future_egomotion = model_config['predict_future_egomotion']  # False
        self.temporal_model_name = model_config['temporal_model_name']  # gru
        self.start_out_channels = model_config['start_out_channels']  # 80
        self.extra_in_channels = model_config['extra_in_channels']  # 8
        self.use_pyramid_pooling = model_config['use_pyramid_pooling']  # False

        self.temporal_model = TemporalModel(
            in_channels=self.camC, receptive_field=self.receptive_field, name=self.temporal_model_name,
        )

        if self.temporal_model_name == 'gru':
            future_pred_in_channels = self.camC
        else:
            future_pred_in_channels = (self.start_out_channels
                                       + self.extra_in_channels * (self.receptive_field - 2))

        self.future_prediction = FuturePrediction(
            in_channels=future_pred_in_channels, latent_dim=self.latent_dim,
            predict_future_egomotion=self.predict_future_egomotion,
        )

        self.bevencode = BevEncode(inC=future_pred_in_channels, outC=outC)

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, future_egomotions):
        b, s, n, c, h, w = imgs.shape
        # Reshape
        imgs = self.pack_sequence_dim(imgs)
        rots = self.pack_sequence_dim(rots)
        trans = self.pack_sequence_dim(trans)
        intrins = self.pack_sequence_dim(intrins)
        post_rots = self.pack_sequence_dim(post_rots)
        post_trans = self.pack_sequence_dim(post_trans)

        # Lifting features
        x = self.get_voxels(imgs, rots, trans, intrins, post_rots, post_trans)
        x = self.unpack_sequence_dim(x, b, s)

        # Temporal model
        z = self.temporal_model(x)

        # Future prediction
        hidden_state = z[:, (self.receptive_field - 1)]  # Only take the present element

        b, _, h, w = hidden_state.shape
        latent_tensor = hidden_state.new_zeros(b, self.n_future, self.latent_dim, h, w)

        z_future = self.future_prediction(latent_tensor, hidden_state,
                                          future_egomotions[:, (self.receptive_field - 1):-1])  # shape (b, n_future,
        # 256,
        # 10, 24)

        # Decode present
        z_t = hidden_state.unsqueeze(1)
        z_future = torch.cat([z_t, z_future], dim=1)

        b, new_s = z_future.shape[:2]

        z_future = self.pack_sequence_dim(z_future)

        # Predict bev segmentations
        bev_output = self.bevencode(z_future)
        bev_output = self.unpack_sequence_dim(bev_output, b, new_s)
        return bev_output


def compile_model(grid_conf, data_aug_conf, outC, name='basic', model_config={}):
    if name == 'basic':
        return LiftSplatShoot(grid_conf, data_aug_conf, outC)
    elif name == 'temporal':
        return TemporalLiftSplatShoot(grid_conf, data_aug_conf, outC, model_config)
