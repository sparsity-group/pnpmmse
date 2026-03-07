import torch
import torch.nn.functional as F

def gaussian_kernel2d(size=5, sigma=2.0, device="cpu"):
    """Return a 2D Gaussian kernel normalized to sum=1."""
    ax = torch.arange(size, device=device) - (size - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma)**2)
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d

class GaussianBlurOp:
    def __init__(self, channels=1, size=7, sigma=2.0, device="cpu"):
        k = gaussian_kernel2d(size, sigma, device)
        self.k = k[None, None, :, :]  # shape (1,1,H,W)
        self.kT = torch.flip(self.k, dims=[-1, -2])  # adjoint = flipped kernel
        self.channels = channels
        self.device = device

    def _pad(self, x):
        kH, kW = self.k.shape[-2:]
        pad = (kW//2, kW//2, kH//2, kH//2)
        return F.pad(x, pad, mode="circular")

    def A(self, x):
        k = self.k.repeat(self.channels, 1, 1, 1)
        return .99*F.conv2d(self._pad(x), k, groups=self.channels)


if __name__ == "__main__":
    print(gaussian_kernel2d())
