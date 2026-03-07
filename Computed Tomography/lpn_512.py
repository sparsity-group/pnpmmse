import numpy as np
import torch
from torch import nn

HIDDEN = 64

class LPN(nn.Module):
    def __init__(self, in_dim, beta):
        super().__init__()

        self.lin = nn.ModuleList([
            nn.Conv2d(in_dim, HIDDEN, 3, bias=True,  stride=1, padding=1),  # 512
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=2, padding=1),  # 256
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=1, padding=1),  # 256
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=2, padding=1),  # 128
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=1, padding=1),  # 128
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=2, padding=1),  # 64
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=1, padding=1),  # 64
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=2, padding=1),  # 32
            nn.Conv2d(HIDDEN,  HIDDEN, 3, bias=False, stride=2, padding=1),  # 16
            nn.Conv2d(HIDDEN,  64,     16, bias=False, stride=1, padding=0), # 1
            nn.Linear(64, 1),
        ])

        self.res = nn.ModuleList([
            nn.Conv2d(in_dim, HIDDEN, 3, stride=2, padding=1),  # 256
            nn.Conv2d(in_dim, HIDDEN, 3, stride=1, padding=1),  # 256
            nn.Conv2d(in_dim, HIDDEN, 3, stride=2, padding=1),  # 128
            nn.Conv2d(in_dim, HIDDEN, 3, stride=1, padding=1),  # 128
            nn.Conv2d(in_dim, HIDDEN, 3, stride=2, padding=1),  # 64
            nn.Conv2d(in_dim, HIDDEN, 3, stride=1, padding=1),  # 64
            nn.Conv2d(in_dim, HIDDEN, 3, stride=2, padding=1),  # 32
            nn.Conv2d(in_dim, HIDDEN, 3, stride=2, padding=1),  # 16
            nn.Conv2d(in_dim, 64,     16, stride=1, padding=0), # 1
        ])

        self.act = nn.Softplus(beta=beta)

    def scalar(self, x):
        bsize = x.shape[0]
        assert x.shape[-1] == x.shape[-2]
        y = self.act(self.lin[0](x.clone()))

        sizes = [512, 256, 256, 128, 128, 64, 64, 32, 16]

        for core, res, sz in zip(self.lin[1:-2], self.res[:-1], sizes[:-1]):
            x_scaled = nn.functional.interpolate(x, (sz, sz), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))

        x_scaled = nn.functional.interpolate(x, (sizes[-1], sizes[-1]), mode="bilinear")
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)

        assert y.shape[2] == y.shape[3] == 1
        y = torch.mean(y, dim=(2, 3))
        y = y.reshape(bsize, 64)
        y = self.lin[-1](y)

        return y

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    def forward(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.scalar(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]
        return grad

    def apply_numpy(self, x):
        assert x.shape[:2] == (512, 512)
        device = next(self.parameters()).device
        x_dim = len(x.shape)
        if x_dim == 2:
            x = x[:, :, np.newaxis]
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            x = self(x)
        x = x[0].detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        if x_dim == 2:
            x = np.squeeze(x, 2)
        return x