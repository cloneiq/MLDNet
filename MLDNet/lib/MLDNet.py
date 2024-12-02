from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.CPCA import CPCA
from lib.SSFA import SSFA
from lib.MPDF import MPDF


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class multi_level_detail_injection(nn.Module):
    def __init__(self, in_c, out_c):  # [320,128,64],64
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_2_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c3 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)
        self.c_msdi = conv2d(in_c[2], out_c, kernel_size=1, padding=0)

        self.c12_11 = conv2d(out_c * 2, out_c)
        self.c12_12 = conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c12_21 = conv2d(out_c * 2, out_c)
        self.c12_22 = conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c4_11 = conv2d(out_c,out_c*2)
        self.c4_12 = conv2d(out_c*2, out_c, kernel_size=1, padding=0)

        self.c4_21 = conv2d(out_c,out_c)
        self.c4_22 = conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.c22 = conv2d(out_c, out_c)
        self.c23 = conv2d(out_c, out_c)

        self.proj = nn.Conv2d(out_c,out_c,1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x1, x2, x3, mpdf):
        x1 = self.up_1(x1)  # 320*22*22 -> 320*88*88
        x1 = self.c1(x1)  # 320*88*88 -> 64*88*88

        x2 = self.c2(x2)  # 128*44*44 -> 64*44*44
        x2 = self.up_2(x2)  # 64*44*44  —> 64*88*88

        x12 = torch.cat([x1, x2], 1)  # 64*88*88  ->  128*88*88
        x12 = self.up_2_1(x12)  # 128*88*88 -> 128*176*176

        x12_1 = self.c12_11(x12)  # 128*176*176 -> 64*176*176
        x12_1 = self.c12_12(x12_1)  # 64*176*176

        x3 = self.up_3(x3)  # 64*88*88 -> 64*176*176
        x3_1 = self.c3(x3)  # 64*176*176xc
        x3_1 = (x3_1 * x12_1)   # 64*176*176

        x_mpdf = self.c_msdi(mpdf)
        x_mpdf = self.up_3(x_mpdf)

        x3_1_sig = torch.sigmoid(x3_1)
        g_att_2 = x3_1_sig * x_mpdf
        x_mpdf_sig = torch.sigmoid(x_mpdf)
        x_att_2 = x_mpdf_sig * x3_1
        interaction = x_att_2 * g_att_2

        x = self.c23(self.c22(interaction))  # 64*176*176
        return x

class MLDNet(nn.Module):
    def __init__(self, channel=32):
        super(MLDNet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.cpca1 = CPCA(64, 64)
        self.cpca2 = CPCA(128, 128)
        self.cpca3 = CPCA(320, 320)
        self.cpca4 = CPCA(512, 512)


        self.ssfa3 = SSFA(320,320)
        self.ssfa2 = SSFA(128,128)
        self.ssfa1 = SSFA(64,64)
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_3 = conv2d(512,320,1,0,1,True)
        self.conv_2 = conv2d(320,128,1,0,1,True)
        self.conv_1 = conv2d(128,64,1,0,1,True)

        self.sigmod = nn.Sigmoid()

        self.mpdf_1 = MPDF([64, 128, 320, 512],0)
        self.mpdf_2 = MPDF([64, 128, 320, 512], 1)
        self.mpdf_3 = MPDF([64, 128, 320, 512], 2)
        self.mpdf_4 = MPDF([64, 128, 320, 512], 3)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.MDI = multi_level_detail_injection([320, 128, 64], 64)

        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(128, 1, 1)
        self.out3 = nn.Conv2d(320, 1, 1)
        self.out4 = nn.Conv2d(512, 1, 1)
        self.out5 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        pvt = self.backbone(x)
        x_1 = pvt[0]
        x_2 = pvt[1]
        x_3 = pvt[2]
        x_4 = pvt[3]

        x_1 = self.cpca1(x_1)
        x_2 = self.cpca2(x_2)
        x_3 = self.cpca3(x_3)
        x_4 = self.cpca4(x_4)


        inputs = [x_1,x_2,x_3,x_4]
        mpdf_1 = self.mpdf_1(inputs,0)    # (H/4,W/4,64)
        mpdf_2 = self.mpdf_2(inputs,1)
        mpdf_3 = self.mpdf_3(inputs,2)
        mpdf_4 = self.mpdf_4(inputs,3)

        x_pr_3_1 = self.ssfa3(self.upsample_3(self.conv_3(mpdf_4)),mpdf_3)
        x_pr_2_1 = self.ssfa2(self.upsample_2(self.conv_2(x_pr_3_1)),mpdf_2)
        x_pr_1_1 = self.ssfa1(self.upsample_1(self.conv_1(x_pr_2_1)),mpdf_1)

        x_out = self.MDI(x_pr_3_1, x_pr_2_1, x_pr_1_1,mpdf_1)

        x_out = F.interpolate(self.out5(x_out), scale_factor=2, mode='bilinear')
        prediction1_4 = F.interpolate(self.out1(x_pr_1_1), scale_factor=4, mode='bilinear')
        prediction2_8 = F.interpolate(self.out2(x_pr_2_1), scale_factor=8, mode='bilinear')
        prediction3_16 = F.interpolate(self.out3(x_pr_3_1), scale_factor=16, mode='bilinear')
        prediction4_32 = F.interpolate(self.out4(mpdf_4), scale_factor=32, mode='bilinear')

        return x_out, prediction1_4, prediction2_8, prediction3_16, prediction4_32


