import torch
import torch.nn as nn
import torch.nn.functional as F

class GoConv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 bn=True):
        super(GoConv3D, self).__init__()

        assert kernel_size[1] == kernel_size[2]
        assert stride[1] == stride[2]
        assert padding[1] == padding[2]

        self.kernel_size = kernel_size[2]
        self.stride = stride[2]
        self.conv3d = nn.Conv3d(in_channels * self.kernel_size * self.kernel_size,
                                out_channels,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=(padding[0], 0, 0),
                                stride=(stride[0], 1, 1),
                                bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        else:
            self.bn = None

    def get_gcacost(self, x_ori, dp, normal, intri):

        kernel_size = self.kernel_size
        stride = self.stride
        b, c, d, h, w = x_ori.shape
       
        h = h // stride
        w = w // stride

        pad = (kernel_size - 1) // 2


        dp_unfold = F.pad(dp, pad=[pad, pad, pad, pad], mode='replicate')
        dp_unfold = F.unfold(dp_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        dp_unfold = dp_unfold.view(b, d, kernel_size ** 2, h, w).squeeze(1)  # b d k2 h w

        normal_p_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_p_unfold = F.unfold(normal_p_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        normal_p_unfold = normal_p_unfold.view(b, 3, kernel_size ** 2, h, w).squeeze(1)

        nx, ny, nz = normal_p_unfold[:, 0, ...], normal_p_unfold[:, 1, ...], normal_p_unfold[:, 2, ...]
        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        h_ori = h * stride
        w_ori = w * stride
        y, x = torch.meshgrid([torch.arange(0, h_ori, dtype=torch.float32, device=normal.device),
                               torch.arange(0, w_ori, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ori * w_ori), x.view(h_ori * w_ori)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(b, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx) / fx  # b h*w
        posy = (xy[:, 1, :] - cy) / fy
        pos = torch.stack([posx, posy], dim=1).reshape(b, 2, h_ori, w_ori)
        pos_p = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_p, [kernel_size, kernel_size], padding=0, stride=stride)
        pos_unfold = pos_unfold.view(b, 2, kernel_size * kernel_size, h, w)
        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]

        # pos - center
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w

        dp_prog = ddw_weights.unsqueeze(1) * dp_unfold[:, :, (kernel_size * kernel_size) // 2, :, :].unsqueeze(
            2)  # b d k2 h w
        interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0] # b k2 h w

        # b d k2 h w  - b 1 k2 h w  / B  1 K*K H W = b d k*k h w
        indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k2 h w
        #indices = indices / ((d - 1) / 2) - 1 # b d k2 h w
        indices = indices.reshape(b, d * kernel_size * kernel_size, h, w).unsqueeze(-1)

        xy_n = xy.reshape(1, 2, h_ori, w_ori)# b 2 h w .permute(0, 2, 1).reshape(0, h_ori, w_ori, 2).unsqueeze(1)  # b 1 h w 2

        xy_p = F.pad(xy_n, pad=[pad, pad, pad, pad], mode='replicate')
        xy_p = F.unfold(xy_p, [kernel_size, kernel_size], padding=0, stride=stride)  # (B, ps*ps, H*W)
        xy_p = xy_p.reshape(b, 2, kernel_size**2, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k2 h w
        xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size * kernel_size, h, w, 2)

        indices = (torch.cat([xy_p,indices], dim=-1))
        indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
        indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
        indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1
        ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
        ans = ans.reshape(b, c, d, kernel_size*kernel_size,h,w).permute(0,1,3,2,4,5)
        return ans.reshape(b, c * kernel_size * kernel_size, d, h, w)

    def forward(self, x, dp, normal, intri):
        x = self.get_gcacost(x, dp, normal, intri)  # b c*k*k, d h w
        x = self.conv3d(x)  # b c,d h w
        if self.bn is not None:
            x = F.relu(self.bn(x), inplace=True)
        return x


class GoUpConv3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 bn=True):
        super(GoUpConv3D, self).__init__()

        assert kernel_size[1] == kernel_size[2]
        assert stride[1] == stride[2]
        assert padding[1] == padding[2]

        self.kernel_size = kernel_size[2]
        self.stride = stride[2]
        self.out_channels = out_channels
        self.conv3d = nn.Conv3d(in_channels * self.kernel_size * self.kernel_size,
                                out_channels * 2 * 2,
                                kernel_size=(kernel_size[0], 1, 1),
                                padding=(padding[0], 0, 0),
                                stride=(stride[0], 1, 1),
                                bias=False)
        if bn:
            self.bn = nn.BatchNorm3d(out_channels, momentum=0.1)
        else:
            self.bn = None

    def get_pecost(self, x_ori, dp, normal, intri):
        # intric b 3 3
        kernel_size = self.kernel_size
        stride = self.stride
        b, c, d, h, w = x_ori.shape
        
        h = h // stride
        w = w // stride

        pad = (kernel_size - 1) // 2

        dp_unfold = F.pad(dp, pad=[pad, pad, pad, pad], mode='replicate')
        dp_unfold = F.unfold(dp_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        dp_unfold = dp_unfold.view(b, d, kernel_size ** 2, h, w).squeeze(1)  # b d k2 h w

        normal_p_unfold = F.pad(normal, pad=[pad, pad, pad, pad], mode='replicate')
        normal_p_unfold = F.unfold(normal_p_unfold, [kernel_size, kernel_size], padding=0, stride=stride)
        normal_p_unfold = normal_p_unfold.view(b, 3, kernel_size ** 2, h, w).squeeze(1)

        nx, ny, nz = normal_p_unfold[:, 0, ...], normal_p_unfold[:, 1, ...], normal_p_unfold[:, 2, ...]
        # (B, 3, ps*ps, H, W)
        fx, fy, cx, cy = intri[:, 0, 0], intri[:, 1, 1], intri[:, 0, 2], intri[:, 1, 2]  # B,
        h_ori = h * stride
        w_ori = w * stride
        y, x = torch.meshgrid([torch.arange(0, h_ori, dtype=torch.float32, device=normal.device),
                               torch.arange(0, w_ori, dtype=torch.float32, device=normal.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(h_ori * w_ori), x.view(h_ori * w_ori)
        xy = torch.stack((x, y))  # [2, H*W]
        xy = xy.unsqueeze(0).repeat(b, 1, 1)  # b 2 h*w

        posx = (xy[:, 0, :] - cx) / fx  # b h*w
        posy = (xy[:, 1, :] - cy) / fy
        pos = torch.stack([posx, posy], dim=1).reshape(b, 2, h_ori, w_ori)
        # print('pos,',pos)
        pos_p = F.pad(pos, pad=[pad, pad, pad, pad], mode='replicate')
        pos_unfold = F.unfold(pos_p, [kernel_size, kernel_size], padding=0, stride=stride)
        # print('pos_unfold,',pos_unfold)
        pos_unfold = pos_unfold.view(b, 2, kernel_size * kernel_size, h, w)

        pos_u, pos_v = pos_unfold[:, 0, ...], pos_unfold[:, 1, ...]

        # pos - center
        pos_u_center = pos_unfold[:, 0, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)
        pos_v_center = pos_unfold[:, 1, (kernel_size * kernel_size) // 2, :, :].unsqueeze(1)

        ddw_num = nx * pos_u_center + ny * pos_v_center + nz  # b k*k h w
        ddw_denom = nx * pos_u + ny * pos_v + nz  # b k*k h w
        ddw_denom[torch.abs(ddw_denom) < 1e-8] = torch.sign(ddw_denom[torch.abs(ddw_denom) < 1e-8]) * 1e-8

        ddw_weights = ddw_num / ddw_denom  # (B, ps*ps, H, W)
        ddw_weights[ddw_weights != ddw_weights] = 1.0  # nan
        ddw_weights[torch.abs(ddw_weights) == float("Inf")] = 1.0  # inf  #b k2 h w

        dp_prog = ddw_weights.unsqueeze(1) * dp_unfold[:, :, (kernel_size * kernel_size) // 2, :, :].unsqueeze(
            2)  # b d k2 h w
        interval_unfold = dp_unfold[:, 1] - dp_unfold[:, 0]  # b k2 h w


        indices = (dp_prog - dp_unfold[:, 0].unsqueeze(1)) / interval_unfold.unsqueeze(1)  # b d k2 h w
        indices = indices.reshape(b, d * kernel_size * kernel_size, h, w).unsqueeze(-1)

        xy_n = xy.reshape(1, 2, h_ori,
                          w_ori) 

        xy_p = F.pad(xy_n, pad=[pad, pad, pad, pad], mode='replicate')
        xy_p = F.unfold(xy_p, [kernel_size, kernel_size], padding=0, stride=stride)  # (B, ps*ps, H*W)
        xy_p = xy_p.reshape(b, 2, kernel_size ** 2, h, w).unsqueeze(1).repeat(1, d, 1, 1, 1, 1)  # b d 2 k2 h w
        xy_p = xy_p.permute(0, 1, 3, 4, 5, 2).reshape(b, d * kernel_size * kernel_size, h, w, 2)

        indices = (torch.cat([xy_p, indices], dim=-1))
        indices[:, :, :, :, 2] = indices[:, :, :, :, 2] / ((d - 1) / 2) - 1
        indices[:, :, :, :, 0] = indices[:, :, :, :, 0] / ((w_ori - 1) / 2) - 1
        indices[:, :, :, :, 1] = indices[:, :, :, :, 1] / ((h_ori - 1) / 2) - 1  # b d*k2 h w 3
        ans = F.grid_sample(x_ori, grid=indices, padding_mode='zeros', mode='bilinear', align_corners=True)
        ans = ans.reshape(b, c, d, kernel_size * kernel_size, h, w).permute(0, 1, 3, 2, 4, 5)
        return ans.reshape(b, c * kernel_size * kernel_size, d, h, w)

    def forward(self, x, pd, normal, intri):

        b, c, d, h, w = x.shape
        x = self.get_pecost(x, pd, normal, intri)  # b c*k*k d h w
        x = self.conv3d(x)  # b c,d h w
        x = x.reshape(b, self.out_channels, 2, 2, d, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, self.out_channels, d, h * 2,w * 2)
        x = F.relu(self.bn(x), inplace=True)
        return x


class GCACostRegNet(nn.Module):
    '''
    input b d h w
    output b d h w
    '''

    def __init__(self, in_channels, base_channels):
        super(GCACostRegNet, self).__init__()

        self.conv0 = GoConv3D(in_channels, base_channels)

        self.conv1 = GoConv3D(base_channels, base_channels * 2, stride=(1, 2, 2))
        self.conv2 = GoConv3D(base_channels * 2, base_channels * 2)

        self.conv3 = GoConv3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2))
        self.conv4 = GoConv3D(base_channels * 4, base_channels * 4)

        self.conv5 = GoConv3D(base_channels * 4, base_channels * 8, stride=(1, 2, 2))
        self.conv6 = GoConv3D(base_channels * 8, base_channels * 8)

        self.conv7 = GoUpConv3D(base_channels * 8, base_channels * 4)

        self.conv9 = GoUpConv3D(base_channels * 4, base_channels * 2)

        self.conv11 = GoUpConv3D(base_channels * 2, base_channels * 1)

        self.prob = GoConv3D(base_channels, 1, bn=False)

    def forward(self, x, d, normal, intri):
        conv0 = self.conv0(x, d, normal, intri)

        d1 = F.interpolate(d, scale_factor=0.5)
        normal1 = F.interpolate(normal, scale_factor=0.5)
        intri1 = intri[:, 0:2, :] * 0.5

        conv2 = self.conv2(self.conv1(conv0, d, normal, intri), d1, normal1, intri1)

        d2 = F.interpolate(d1, scale_factor=0.5)
        normal2 = F.interpolate(normal1, scale_factor=0.5)
        intri2 = intri1[:, 0:2, :] * 0.5

        conv4 = self.conv4(self.conv3(conv2, d1, normal1, intri1), d2, normal2, intri2)

        d3 = F.interpolate(d2, scale_factor=0.5)
        normal3 = F.interpolate(normal2, scale_factor=0.5)
        intri3 = intri2[:, 0:2, :] * 0.5

        x = self.conv6(self.conv5(conv4, d2, normal2, intri2), d3, normal3, intri3)  # b 32 d h/8 w/8
        x = conv4 + self.conv7(x, d3, normal3, intri3)  # b 16 d h/4 w/4
        x = conv2 + self.conv9(x, d2, normal2, intri2)  # b 8 d h/2 w/2
        x = conv0 + self.conv11(x, d1, normal1, intri1)  # b 4 d h w
        x = self.prob(x, d, normal, intri)
        return x
