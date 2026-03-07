import torch
from torch.distributions import MixtureSameFamily, Categorical, Normal


class Sampler:
    def __init__(self, weights, means, stds, type="Gaussian", device=None, dtype=torch.float32):
        self.device = device
        self.dtype  = dtype

        weights = torch.as_tensor(weights, device=device, dtype=dtype)
        means   = torch.as_tensor(means,   device=device, dtype=dtype)
        stds    = torch.as_tensor(stds,    device=device, dtype=dtype)

        if not torch.isclose(weights.sum(), torch.tensor(1., dtype=dtype, device=device)):
            weights = weights / weights.sum()

        cat = torch.distributions.Categorical(weights)
        if type == "Gaussian":
            comps = torch.distributions.Normal(means, stds)
        elif type == "Laplacian":
            comps = torch.distributions.Laplace(means, stds)
        else:
            raise ValueError("Unknown distribution type!")

        self.dist = torch.distributions.MixtureSameFamily(cat, comps)

    def __call__(self, n):
        return self.dist.sample((n,))


import torch.nn.functional as F


def convolve_with_unit_gaussian(f_X, x):
    """
    Numerically convolve exp(-f_X) with a unit Gaussian, return -log of result.
    """
    dx = x[1] - x[0]
    p_X = (-f_X).exp()

    # Unit Gaussian kernel on same grid
    gauss = torch.exp(-0.5 * x ** 2) / (2 * torch.pi) ** 0.5

    # Convolve via FFT
    n = len(x)
    conv = torch.fft.ifftshift(
        torch.fft.irfft(
            torch.fft.rfft(p_X, n=2 * n) * torch.fft.rfft(gauss, n=2 * n),
            n=2 * n
        )
    )[:n] * dx

    return -conv.log()
