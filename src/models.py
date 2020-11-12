"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from src.layers.convolutions import ResBlock, ConvBlock, Bottleneck
from src.layers.temporal import SpatialGRU, Bottleneck3D, TemporalBlock
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum, pose_vec2mat
from .constants import MIN_LOG_SIGMA, MAX_LOG_SIGMA


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
    def __init__(self, inC, outC, output_cost_map=False):
        super(BevEncode, self).__init__()
        self.output_cost_map = output_cost_map

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

        if self.output_cost_map:
            self.cost_map_decoder = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        output['bev'] = self.up2(x)
        if self.output_cost_map:
            output['cost_map'] = self.cost_map_decoder(x)

        return output


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
        return {'bev': x['bev']}


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
            return nn.Sequential(SpatialGRU(self.in_channels, self.in_channels, norm=self.norm,
                                            activation=self.activation),
                                 SpatialGRU(self.in_channels, self.in_channels, norm=self.norm,
                                            activation=self.activation),
                                 SpatialGRU(self.in_channels, self.in_channels, norm=self.norm,
                                            activation=self.activation),
                                 )
        elif self.name == 'temporal_block':
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
        elif self.name == 'identity':
            return nn.Sequential()

    def forward(self, x):
        if self.name == 'gru':
            return self.model(x)
        elif self.name == 'temporal_block':
            # Reshape input tensor to (batch, C, time, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            z = self.model(x)
            return z.permute(0, 2, 1, 3, 4).contiguous()
        else:
            return x

    @property
    def receptive_field(self):
        return self._receptive_field


class DistributionEncoder(nn.Module):
    """Encodes z_t and (z_{t+1}, z_{t+2}, ..., z_{t+N}) with z_t the dynamics features, and N the number of
    predicted future frames.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        future_distribution=False,
        future_in_channels=None,
        n_future=None,
    ):
        super().__init__()
        self.future_distribution = future_distribution
        self.n_future = n_future

        encoder_in_channels = in_channels
        if future_distribution:
            self.future_compression = Bottleneck(
                future_in_channels, out_channels=in_channels, kernel_size=1,
            )
            encoder_in_channels += in_channels

        self._encoder = nn.Sequential(
            Bottleneck(encoder_in_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
            Bottleneck(out_channels, out_channels=out_channels, downsample=True),
        )

    def forward(self, z_t, z_future=None):
        output = []
        b, s, _, h, w = z_t.shape

        for t in range(s):
            input_t = z_t[:, t]
            if self.future_distribution:
                compression_input = z_future.view(b, -1, h, w)
                input_future_t = self.future_compression(compression_input)
                input_t = torch.cat([input_t, input_future_t], dim=1)

            output.append(self._encoder(input_t))

        return self.pack_sequence_dim(torch.stack(output, dim=1))

    @staticmethod
    def pack_sequence_dim(x):
        b, s = x.shape[:2]
        return x.view(b * s, *x.shape[2:])


class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
        self,
        in_channels,
        latent_dim,
        future_distribution=False,
        future_in_channels=None,
        n_future=None,
    ):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.n_future = n_future
        self.encoder = DistributionEncoder(
            in_channels,
            self.compress_dim,
            future_distribution=future_distribution,
            future_in_channels=future_in_channels,
            n_future=n_future,
        )
        # TODO: Apply Average pooling to each z_{t+i} feature before concatenating.
        self.out_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
        )

    def forward(self, z_t, z_future=None):
        b, s = z_t.shape[:2]
        assert s == 1
        encoding = self.encoder(z_t, z_future)

        mu_log_sigma = self.out_module(encoding).view(b, s, 2 * self.latent_dim)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(log_sigma, MIN_LOG_SIGMA, MAX_LOG_SIGMA)
        return mu, log_sigma


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

    def forward(self, x, hidden_state, future_egomotions, inference=False, pose_net=None,
                direct_trajectory_prediction=False):
        # pylint: disable=arguments-differ
        # x has shape (b, n_future, c, h, w), hidden_state (b, c, h, w)
        if not inference:
            for i in range(self.n_gru_blocks):
                if self.predict_future_egomotion:
                    # Warp features with respect to future ego-motion
                    flow = future_egomotions
                else:
                    flow = None

                x = self.spatial_grus[i](x, hidden_state, flow)
                b, n_future, c, h, w = x.shape

                x = self.res_blocks[i](x.view(b * n_future, c, h, w))
                x = x.view(b, n_future, c, h, w)

            return x, None

        else:
            # Use predicted future ego-motion to warp features in a recursive manner
            output = []
            pred_future_egomotions = []
            intermediate_states = {}
            n_future = x.shape[1]

            initial_hidden_state = hidden_state

            # This is our Markovian state
            hidden_state_t = initial_hidden_state

            if direct_trajectory_prediction:
                # z_t + sample
                full_flow = pose_net(torch.cat([hidden_state_t, x[:, 0]], dim=1))
            for t in range(n_future):
                # Compute future egomotion
                if self.predict_future_egomotion:
                    if direct_trajectory_prediction:
                        flow_t = full_flow[:, t]
                    else:
                        flow_t = pose_net(hidden_state_t)
                        pred_future_egomotions.append(flow_t)
                else:
                    flow_t = None
                for i in range(self.n_gru_blocks):
                    if i == 0:
                        gru_input = x[:, t]
                    else:
                        gru_input = hidden_state_next
                    if t == 0:
                        hidden_state_previous = initial_hidden_state
                    else:
                        hidden_state_previous = intermediate_states[(t-1, i)]

                    # Warp
                    hidden_state_previous = self.spatial_grus[i].warp_features(hidden_state_previous, flow_t)

                    hidden_state_next = self.spatial_grus[i].gru_cell(gru_input, hidden_state_previous)

                    intermediate_states[(t, i)] = hidden_state_next

                    hidden_state_next = self.res_blocks[i](hidden_state_next)

                # New markovian state at the end of all the gru forwards
                hidden_state_t = hidden_state_next

                output.append(hidden_state_t)

            output = torch.stack(output, dim=1)

            if self.predict_future_egomotion:
                if direct_trajectory_prediction:
                    pred_future_egomotions = full_flow
                else:
                    flow_t = pose_net(hidden_state_t)
                    pred_future_egomotions.append(flow_t)
                    pred_future_egomotions = torch.stack(pred_future_egomotions, dim=1)
            else:
                pred_future_egomotions = None

            return output, pred_future_egomotions


class FuturePredictionAutoregressive(torch.nn.Module):
    def __init__(self, in_channels, latent_dim, predict_future_egomotion=True, n_gru_blocks=3, n_res_layers=3):
        super().__init__()
        self.predict_future_egomotion = predict_future_egomotion
        self.n_gru_blocks = n_gru_blocks

        # Transform the input from latent_dim to in_channels
        self.conv_match_channel_hidden_state = nn.Conv2d(latent_dim, in_channels, kernel_size=1)

        # Convolutional recurrent model with z_t as an initial hidden state and inputs the sample
        # from the probabilistic model. The architecture of the model is:
        # [Spatial GRU - [Bottleneck] x n_res_layers] x n_gru_blocks
        self.spatial_grus = []
        self.res_blocks = []

        for i in range(self.n_gru_blocks):
            hidden_size = in_channels
            self.spatial_grus.append(SpatialGRU(in_channels, hidden_size, autoregressive=True))
            self.res_blocks.append(torch.nn.Sequential(*[ResBlock(in_channels)
                                                         for _ in range(n_res_layers)]))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)

    def forward(self, z, hidden_state, future_egomotions, inference=False, pose_net=None,
                direct_trajectory_prediction=False):
        # pylint: disable=arguments-differ
        # z has shape (b, n_future, c, h, w), hidden_state (b, c, h, w) and is the sample from the distribution
        if not inference:
            hidden_state = self.conv_match_channel_hidden_state(hidden_state)
            for i in range(self.n_gru_blocks):
                if self.predict_future_egomotion and i == 0:
                    # Warp features with respect to future ego-motion
                    flow = future_egomotions
                else:
                    flow = None

                z = self.spatial_grus[i](z, hidden_state, flow)
                b, n_future, c, h, w = z.shape

                z = self.res_blocks[i](z.view(b * n_future, c, h, w))
                z = z.view(b, n_future, c, h, w)

            return z, None
        else:
            # Use predicted future ego-motion to warp features in a recursive manner
            output = []
            pred_future_egomotions = []
            intermediate_states = {}
            n_future = z.shape[1]

            latent_vector = self.conv_match_channel_hidden_state(hidden_state)

            # This is our Markovian state
            z_t = z[:, 0]
            for t in range(n_future):
                for i in range(self.n_gru_blocks):
                    if i == 0:
                        # Compute future egomotion
                        flow_t = pose_net(z_t)
                        pred_future_egomotions.append(flow_t)
                        # Warp features
                        gru_input = self.spatial_grus[i].warp_features(z_t, flow_t)
                    else:
                        gru_input = z_t

                    if t == 0:
                        previous_hidden_state = latent_vector
                    else:
                        previous_hidden_state = intermediate_states[(t-1, i)]

                    z_t = self.spatial_grus[i].gru_cell(gru_input, previous_hidden_state)
                    intermediate_states[(t, i)] = z_t

                    z_t = self.res_blocks[i](z_t)

                # New markovian state at the end of all the gru forwards
                output.append(z_t)

            # Compute pose for last state
            flow_t = pose_net(z_t)
            pred_future_egomotions.append(flow_t)

            output = torch.stack(output, dim=1)
            pred_future_egomotions = torch.stack(pred_future_egomotions, dim=1)

            return output, pred_future_egomotions


class PoseNet(nn.Module):
    def __init__(self, in_channels, n_predictions=1, output_channels=6):
        super().__init__()
        self.n_predictions = n_predictions
        self.output_channels = output_channels

        self.module = nn.Sequential(
            ConvBlock(in_channels, 64, 7, 2),
            ConvBlock(64, 64, 5, 2),
            ConvBlock(64, 64, 3, 2),
            ConvBlock(64, 128, 3, 2),
            ConvBlock(128, 256, 3, 2),
            ConvBlock(256, 256, 3, 2),
            ConvBlock(256, 256, 3, 2),
            nn.Conv2d(256, 6*n_predictions, 1),
        )

    def forward(self, x):
        out = self.module(x)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, 6*self.n_predictions)

        if self.n_predictions > 1:
            out = out.view(-1, self.n_predictions, self.output_channels)

        return out


class TemporalLiftSplatShoot(LiftSplatShoot):
    def __init__(self, grid_conf, data_aug_conf, outC, model_config):
        super().__init__(grid_conf, data_aug_conf, outC)

        self.receptive_field = model_config['receptive_field']  # 3
        self.n_future = model_config['n_future']  # 5
        self.latent_dim = model_config['latent_dim']  # 16
        self.probabilistic = model_config['probabilistic']
        self.autoregressive_future_prediction = model_config['autoregressive_future_prediction']
        self.autoregressive_l2_loss = model_config['autoregressive_l2_loss']
        self.direct_trajectory_prediction = model_config['direct_trajectory_prediction']
        self.predict_future_egomotion = model_config['predict_future_egomotion']
        self.output_cost_map = model_config['output_cost_map']
        self.three_dof_egomotion = model_config['three_dof_egomotion']
        self.temporal_model_name = model_config['temporal_model_name']  # gru
        self.disable_bev_prediction = model_config['disable_bev_prediction']
        self.start_out_channels = model_config['start_out_channels']  # 80
        self.extra_in_channels = model_config['extra_in_channels']  # 8
        self.use_pyramid_pooling = model_config['use_pyramid_pooling']  # False

        self.temporal_model = TemporalModel(
            in_channels=self.camC, receptive_field=self.receptive_field, name=self.temporal_model_name,
        )

        if self.temporal_model_name == 'gru':
            future_pred_in_channels = self.camC
        elif self.temporal_model_name == 'temporal_block':
            future_pred_in_channels = (self.start_out_channels
                                       + self.extra_in_channels * (self.receptive_field - 2))
        elif self.temporal_model_name == 'identity':
            future_pred_in_channels = self.camC
        else:
            raise ValueError(f'Unknown temporal model: {self.temporal_model_name}')

        self.future_pred_in_channels = future_pred_in_channels

        self.future_indices = None
        if self.probabilistic:
            self.present_distribution = DistributionModule(
                self.future_pred_in_channels, self.latent_dim,
            )

            future_indices, future_in_channels = self._calculate_future_indices_and_channels()
            assert future_in_channels > 0
            self.future_indices = future_indices

            self.future_distribution = DistributionModule(
                self.future_pred_in_channels,
                self.latent_dim,
                future_distribution=True,
                future_in_channels=future_in_channels,
                n_future=self.n_future,
            )

        if not self.autoregressive_future_prediction:
            self.future_prediction = FuturePrediction(
                in_channels=self.future_pred_in_channels, latent_dim=self.latent_dim,
                predict_future_egomotion=self.predict_future_egomotion,
            )

        else:
            self.future_prediction = FuturePredictionAutoregressive(
                in_channels=self.future_pred_in_channels, latent_dim=self.latent_dim,
                predict_future_egomotion=self.predict_future_egomotion,
            )

        if not self.disable_bev_prediction:
            self.bevencode = BevEncode(inC=self.future_pred_in_channels, outC=outC, output_cost_map=self.output_cost_map)

        if self.predict_future_egomotion:
            n_predictions = 1
            pose_in_channels = self.future_pred_in_channels
            if self.direct_trajectory_prediction:
                n_predictions += self.n_future
                pose_in_channels += self.latent_dim

            self.pose_net = PoseNet(pose_in_channels, n_predictions=n_predictions)
        else:
            self.pose_net = None


    def _calculate_future_indices_and_channels(self):
        """ Calculates which indices would be used for the future distribution """
        # For example if receptive_field_t=5 and n_future=10, the indices will be [9, 14]
        seq_len = self.receptive_field + self.n_future
        future_indices = list(range(self.receptive_field - 1, seq_len, self.receptive_field))
        future_indices = future_indices[1:]
        future_in_channels = len(future_indices) * self.future_pred_in_channels

        return future_indices, future_in_channels

    def _extract_present_future_features(self, z):
        present_features = z[:, (self.receptive_field - 1):self.receptive_field].contiguous()

        # Future features for the future distribution
        if self.future_indices is None:
            future_features = None
        else:
            future_features = z[:, self.future_indices].contiguous()

        return present_features, future_features

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, future_egomotions, inference=False,
                noise=None):

        output = {}
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

        output['z'] = z

        # Split into present and future features (for the probabilistic model)
        present_features, future_features = self._extract_present_future_features(z)

        if self.probabilistic:
            # Do probabilistic computation
            sample, output_distribution = self.distribution_forward(present_features, future_features, noise)

            output = {**output, **output_distribution}

        # Future prediction

        b, _, _, h, w = present_features.shape
        if not self.autoregressive_future_prediction:
            hidden_state = present_features[:, 0]

            if self.probabilistic:
                future_prediction_input = sample.expand(-1, self.n_future, -1, -1, -1)
            else:
                future_prediction_input = hidden_state.new_zeros(b, self.n_future, self.latent_dim, h, w)
        else:
            if self.probabilistic:
                hidden_state = sample[:, 0]
            else:
                hidden_state = present_features.new_zeros(b, self.latent_dim, h, w)

            future_prediction_input = z[:, (self.receptive_field - 1):-1]

        z_future, pred_future_egomotions = self.future_prediction(
            future_prediction_input, hidden_state, future_egomotions[:, (self.receptive_field - 1):-1],
            inference=inference, pose_net=self.pose_net, direct_trajectory_prediction=self.direct_trajectory_prediction,
        )

        output['z_future_pred'] = z_future

        # Decode present
        z_future = torch.cat([present_features, z_future], dim=1)

        b, new_s = z_future.shape[:2]

        z_future = self.pack_sequence_dim(z_future)

        # Predict bev segmentations
        if not self.disable_bev_prediction:
            bev_output = self.bevencode(z_future)
            for key, value in bev_output.items():
                bev_output[key] = self.unpack_sequence_dim(value, b, new_s)
            output = {**output, **bev_output}

        if self.predict_future_egomotion and not inference:
            if self.direct_trajectory_prediction:
                pose_net_input = torch.cat([present_features, sample], dim=-3)
                pred_future_egomotions = self.pose_net(pose_net_input[:, 0])
            else:
                pred_future_egomotions = self.pose_net(z_future)
                pred_future_egomotions = self.unpack_sequence_dim(pred_future_egomotions, b, new_s)

        output['future_egomotions'] = pred_future_egomotions

        return output

    def distribution_forward(self, present_features, future_features, noise):
        """
        Inputs:
            present_features: 5-D output from dynamics module with shape
            future_features: 5-D output from dynamics module with shape
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns:
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        present_mu, present_log_sigma = None, None
        future_mu, future_log_sigma = None, None
        if self.probabilistic:
            present_mu, present_log_sigma = self.present_distribution(present_features, None)
            future_mu, future_log_sigma = self.future_distribution(present_features, future_features)

            if noise is None:
                if self.training:
                    noise = torch.randn_like(present_mu)
                else:
                    noise = torch.zeros_like(present_mu)
            if self.training:
                mu = future_mu
                sigma = torch.exp(future_log_sigma)
            else:
                mu = present_mu
                sigma = torch.exp(present_log_sigma)
            sample = mu + sigma * noise

            # Spatially broadcast sample to the dimensions of present_features
            sample = sample.view(b, s, self.latent_dim, 1, 1).expand(b, s, self.latent_dim, h, w)
        else:
            sample = None

        output_distribution = {'present_mu': present_mu,
                               'present_log_sigma': present_log_sigma,
                               'future_mu': future_mu,
                               'future_log_sigma': future_log_sigma,
                               }

        return sample, output_distribution


def compile_model(grid_conf, data_aug_conf, outC, name='basic', model_config={}):
    if name == 'basic':
        return LiftSplatShoot(grid_conf, data_aug_conf, outC)
    elif name == 'temporal':
        return TemporalLiftSplatShoot(grid_conf, data_aug_conf, outC, model_config)
