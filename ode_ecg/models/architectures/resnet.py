import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        # [b, 64, 100]
        self.conv1 = torch.nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        # [b, 64, 100]
        self.norm1 = torch.nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=1, stride=1, padding=0)
        self.norm2 = torch.nn.BatchNorm1d(64)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x + shortcut


class ResGenerator(nn.Module):
    def __init__(self, device):
        super(ResGenerator, self).__init__()
        self.z = torch.tensor(np.random.normal(0, 1, (40, 98))).float().to(device)
        self.device = device
        # [b, 1, 100]
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv_1 = torch.nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.residuals = ResBlock()
        self.residuals2 = ResBlock()
        self.bottleneck1 = torch.nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bottleneck2 = torch.nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 1)

    def forward(self, t, x):
        if t == 1:
            print("Initializing latent variable...")
            self.z = torch.tensor(np.random.normal(0, 1, (x.shape[0], 98))).float().to(self.device)
        z = self.z
        t = t.repeat(40, 1)
        t = t.view(-1, 1)
        x = torch.cat([z, x, t], dim=1)
        x = x.view(-1, 1, 100)
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.residuals(x)
        x = self.residuals2(x)
        x = self.bottleneck1(x)
        x = self.relu(x)
        x = self.bottleneck2(x)
        x = self.relu(x)
        x = x.view(-1, 100)
        x = self.fc1(x)
        # x = self.relu(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(DCDiscriminator, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        self.main = nn.Sequential(
        # input is (nc) x 64 x 64
        nn.Conv1d(in_channels=1, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 32 x 32
        nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 16 x 16
        nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 8 x 8
        nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        nn.BatchNorm1d(ndf * 16),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. (ndf*8) x 4 x 4
        nn.Conv1d(ndf * 16, 1, 5, 2, 0, bias=False),
        nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 216)
        return self.main(x)
