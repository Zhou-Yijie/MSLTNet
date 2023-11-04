import torch
import torch.nn.functional as F
from torch import nn
from . import fenet

# 3layers with control
class MLPNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64):
        super(MLPNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        # self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Linear(base_nf, base_nf, bias=True)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        cond1 = self.conv2(out)
        cond1 = torch.mean(cond1, dim=[2, 3], keepdim=False)
        scale1 = self.cond_scale1(cond1)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(self.conv3(out))
        return out

class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1,
                 use_bias=True, activation=nn.PReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size,
                              padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

class GuideNN(nn.Module):
    def __init__(self, bn=True):
        super(GuideNN, self).__init__()

        self.conv1 = ConvBlock(3, 8, kernel_size=3, padding=1, batch_norm=bn)
        self.conv3 = ConvBlock(8, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv3(output)

        return output

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        result = torch.cat([R, G, B], dim=1)

        return result

class B_transformer(nn.Module):
    def __init__(self):
        super(B_transformer, self).__init__()

        self.guide = MLPNet(in_nc=3, out_nc=1, base_nf=8)

        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        #self.u_net = FDADN.FDADN(in_nc=3, nf=64, out_nc=8)
        self.fenet = fenet.HFD(in_nc=3, out_nc=8, base_nf=40)
        # self.u_net_mini = FDADN(in_nc=3, nf=64, out_nc=3)
        # self.u_net_mini = UNet_mini(n_channels=3)
        self.smooth = nn.PReLU()
        self.fitune = MLPNet(in_nc=3, out_nc=3, base_nf=8)

        self.p = nn.PReLU()

        self.point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x_u= F.interpolate(x, (48, 48), mode='bicubic', align_corners=True)
        x_r = F.interpolate(x, (48, 48), mode='bicubic', align_corners=True)
        coeff = self.fenet(x_r).reshape(-1, 12, 6, 16, 16)
        
        guidance = self.guide(x)
        slice_coeffs = self.slice(coeff, guidance)

        # x_u = self.u_net_mini(x_u)
        # x_u = F.interpolate(x_u, (x.shape[2], x.shape[3]), mode='bicubic', align_corners=True
        output = self.apply_coeffs(slice_coeffs, self.p(self.point(x))) 

        output = self.fitune(output)

        return output

if __name__=='__main__':
    for i in range(1000):
        bt = B_transformer().cuda()
        data = torch.zeros(1, 3, 1024, 1024).cuda()
        x = bt(data)
        torch.cuda.synchronize()
        print(bt(data).shape)
