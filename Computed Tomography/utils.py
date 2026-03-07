import importlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import json
from omegaconf import OmegaConf
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim



def get_model(model_config):
    """Load model from config file.
    Parameters:
        model_config (OmegaConf): Model config.
    """
    model = importlib.import_module("lpn.networks." + model_config.model).LPN(
        **model_config.params
    )
    model.init_weights(-10, 0.1)
    return model


def _load_model_helper(model_config, model_path):
    """Helper for loading LPN model for testing"""
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model


def load_model(model_path):
    """Load LPN model for testing"""
    model_config = load_config(
        os.path.join(os.path.dirname(model_path), "model_config.json")
    )
    return _load_model_helper(model_config, model_path)


def get_loss_hparams_and_lr(args, global_step):
    """Get loss hyperparameters and learning rate based on training schedule.
    Parameters:
        args (argparse.Namespace): Arguments from command line.
        global_step (int): Current training step.
    """
    loss_hparams = {"type": "l2"}
    if global_step < 1_000:  # warmup
        lr= 1e-3 * global_step / 1_000
    elif global_step < 10_000:
        lr= 1e-3
    elif global_step < 20_000:
        lr= 3e-4
    elif global_step < 30_000:
        lr= 1e-4
    else:
        lr= 3e-5


    return loss_hparams, lr


def get_loss(loss_hparams):
    """Get loss function from hyperparameters.
    Parameters:
        loss_hparams (dict): Hyperparameters for loss function.
    """
    if loss_hparams["type"] == "l1":
        return nn.L1Loss()
    elif loss_hparams["type"] == "prox_matching":
        return ExpDiracSrgt(sigma=loss_hparams["sigma"])
    elif loss_hparams["type"] == "l2":
        return nn.MSELoss()
    else:
        raise NotImplementedError


# surrogate L0 loss: -exp(-(x/sigma)^2) + 1
def exp_func(x, sigma):
    return -torch.exp(-((x / sigma) ** 2)) + 1


class ExpDiracSrgt(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, input, target):
        """
        input, target: batch, *
        """
        bsize = input.shape[0]
        dist = (input - target).pow(2).reshape(bsize, -1).sum(1).sqrt()
        return exp_func(dist, self.sigma).mean()


def center_crop(img, shape):
    """Center crop image to desired shape.
    Args:
        img: image to be cropped, (h, w, c), numpy array
        shape: desired shape, (h, w)
    Returns:
        img_crop: cropped image, (h, w, c), numpy array
    """
    h, w = img.shape[:2]
    h1, w1 = shape
    assert (h - h1) % 2 == 0 and (w - w1) % 2 == 0
    h_start = (h - h1) // 2
    w_start = (w - w1) // 2
    img_crop = img[h_start : h_start + h1, w_start : w_start + w1, ...]
    return img_crop



def get_mayoct(config):
    """Get MayoCT images"""
    dataset = MayoCTDataset(root=config.root, split=config.split)
    x_list = []
    for idx in range(config.start_idx, config.start_idx + config.num_imgs):
        img = dataset[idx]["image"]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # (c, h, w) -> (h, w, c)
        if config.squeeze:
            img = np.squeeze(img, 2)
        x_list.append(img)
    return x_list


def load_config(config_path):
    if config_path is None:
        return None
    with open(config_path, "r") as f:
        config = json.load(f)
    config = OmegaConf.create(config)
    return config


class MayoCTDataset(Dataset):
    def __init__(self, root, split):
        self.data_dir = os.path.join(root, "mayo_data_arranged_patientwise", split, "Phantom")
        self.files = sorted(os.listdir(self.data_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.data_dir, self.files[idx])).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(0)
        return {"image": image}


def load_dataset(root, split):
    return MayoCTDataset(root=root, split=split)


def measure(x_list, A, sigma_noise, seed):
    """Measure images with forward operator and add noise
    Inputs:
        x_list: list of images to be measured
        A: forward operator
        sigma_noise: standard deviation of noise
        seed: random seed
    Outputs:
        y_list: list of measurements
    """
    # set random seed
    np.random.seed(seed)

    y_list = []
    for x in x_list:
        y = A(x)
        noise = np.random.normal(0, 1, y.shape)
        y = y + sigma_noise * noise
        y_list.append(np.asarray(y).astype("float32"))
    return y_list






def compute_psnr(gt, pred):
    """Compute PSNR between gt and pred.
    Args:
        gt: ground truth image, (h, w, c), numpy array
        pred: predicted image, (h, w, c), numpy array
    Returns:
        psnr: PSNR value
    """
    psnr = skimage_psnr(gt, pred, data_range=1.0)
    return psnr


def compute_ssim(gt, pred):
    """Compute SSIM between gt and pred.
    Args:
        gt: ground truth image, (h, w, c)
        pred: predicted image, (h, w, c)
    Returns:
        ssim: SSIM value
    """
    ssim = skimage_ssim(gt, pred, channel_axis=2, data_range=1.0)
    return ssim

def _to_np_img(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().squeeze().numpy()
    t = np.array(t).squeeze()
    # normalize to [0, 1]
    t = t - t.min()
    if t.max() > 0:
        t = t / t.max()
    return t.astype(np.float32)

def _to_hw3(img):
    # img is [H,W], return [H,W,1] for skimage metrics
    return img[:, :, np.newaxis]

def show_and_save_grid(clean_list, noisy_list, deno_list, out_path="ct_denoise_grid.pdf"):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    rows = len(clean_list)
    cols = 3
    titles = ["Clean", "Noisy", "Denoised"]

    fig_w = 7.16
    per_row_h = fig_w / cols
    fig_h = per_row_h * rows

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=0.05, hspace=0.05)
    )
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0, wspace=0.05, hspace=0.05)

    for r in range(rows):
        clean = _to_np_img(clean_list[r])
        noisy = _to_np_img(noisy_list[r])
        deno  = _to_np_img(deno_list[r])

        panels = [clean, noisy, deno]

        clean3 = _to_hw3(clean)
        noisy3 = _to_hw3(noisy)
        deno3  = _to_hw3(deno)

        metrics = [
            None,
            (compute_psnr(clean3, noisy3), compute_ssim(clean3, noisy3)),
            (compute_psnr(clean3, deno3),  compute_ssim(clean3, deno3)),
        ]

        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(panels[c], cmap="gray", vmin=panels[c].min(), vmax=panels[c].max())
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            ax.axis("off")

            if c > 0:
                psnr, ssim = metrics[c]
                label = f"PSNR {psnr:.2f}dB\nSSIM {ssim:.3f}"
                ax.text(
                    0.02, 0.98, label,
                    transform=ax.transAxes,
                    fontsize=6, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                )

    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=400)
    plt.show()          # ← display inline (Jupyter/IPython)
    plt.close(fig)
    print(f"[viz] Saved grid to: {out_path}")