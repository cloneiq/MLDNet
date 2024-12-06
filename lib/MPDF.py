import torch
import torch.nn as nn
import torch.nn.functional as F

class MPDF(nn.Module):
    def __init__(self, channel, i):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(c, channel[i], kernel_size=3, stride=1, padding=1) for c in channel])

    def forward(self, x, i):
        ans = torch.ones_like(x[i])
        target_size = x[i].shape[-2:]

        for i, x in enumerate(x):
            if x.shape[-1] > target_size[0]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[0]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)
        return ans
