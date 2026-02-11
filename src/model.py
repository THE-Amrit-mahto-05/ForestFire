import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_last = nn.Conv2d(128, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        u1 = self.up(c2)
        if u1.shape != c1.shape:
            u1 = torch.nn.functional.interpolate(u1, size=c1.shape[2:])
        merge = torch.cat([u1, c1], dim=1)
        out = self.conv_last(merge)
        return self.sigmoid(out)