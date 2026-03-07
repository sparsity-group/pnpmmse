import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
# from lpn.utils import metrics
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


# ----------------------------------------------------------------------
# ----------------------------- Data Loader ----------------------------
# ----------------------------------------------------------------------

class MNISTDataset(Dataset):
    """
    Reads data/mnist.npy (and optional data/labels.npy).
    Returns dicts with tensors; avoids per-item conversions.
    Splits: train[0:50000], test[50000:55000], valid[55000:].
    """
    def __init__(self, split: str):
        # ---- load images ----
        imgs = np.load("data/mnist.npy")
        # normalize only if uint8
        if imgs.dtype == np.uint8:
            imgs = imgs.astype(np.float32) / 255.0
        else:
            imgs = imgs.astype(np.float32)

        # ensure (N,1,28,28)
        if imgs.ndim == 3:
            imgs = imgs[:, None, :, :]

        # ---- optional labels ----
        labels_path = "data/labels.npy"
        labels = None
        if os.path.exists(labels_path):
            labels = np.load(labels_path).astype(np.int64)

        # ---- choose split ----
        if split == "train":
            sl = slice(0, 50000)
        elif split == "test":
            sl = slice(50000, 55000)
        elif split == "valid":
            sl = slice(55000, None)
        else:
            raise ValueError(f"Unknown split: {split}")

        # ---- slice once, convert once (zero-copy to torch) ----
        self.x = torch.from_numpy(imgs[sl])                     # (N,1,28,28) float32
        self.y = torch.from_numpy(labels[sl]) if labels is not None else None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self.y is None:
            return {"image": self.x[i]}
        else:
            return {"image": self.x[i], "label": self.y[i]}

# loader with same signature you used
def load_dataset(split):
    ds = MNISTDataset(split=split)
    print(f"dataset loaded: {split}")
    return ds


# ----------------------------------------------------------------------
# ----------------------------- Validation -----------------------------
# ----------------------------------------------------------------------


class Validator:
    """Class for validation."""

    def __init__(self, dataloader, writer, sigma_noise):
        self.dataloader = dataloader
        self.writer = writer
        self.sigma_noise = sigma_noise

    def _validate(self, model):
        """Validate the model on the validation set."""

        model.eval()
        device = next(model.parameters()).device

        psnr_list = []
        ssim_list = []
        for batch in self.dataloader:
            clean_images = batch["image"].to(device)
            noise = torch.randn_like(clean_images)
            noisy_images = clean_images + noise * self.sigma_noise
            out = model(noisy_images)

            psnr_, ssim_ = self.compute_metrics(clean_images, out)
            psnr_list.extend(psnr_)
            ssim_list.extend(ssim_)

        return {
            "PSNR": np.mean(psnr_list),
            "SSIM": np.mean(ssim_list)
        }

    def compute_metrics(self, gt, out):
        """gt, out: batch, channel, height, width. torch.Tensor."""
        gt = gt.cpu().detach().numpy().transpose(0, 2, 3, 1)
        out = out.cpu().detach().numpy().transpose(0, 2, 3, 1)

        psnr_ = [metrics.compute_psnr(gt_, out_) for gt_, out_ in zip(gt, out)]
        ssim_ = [metrics.compute_ssim(gt_, out_) for gt_, out_ in zip(gt, out)]

        return psnr_, ssim_
    def validate(self, model, step):
        """Validate the model and log the metrics."""
        return self._validate(model)


# ----------------------------------------------------------------------
# ------------------------------- metrics ------------------------------
# ----------------------------------------------------------------------


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

def show_and_save_grid(clean_list, noisy_list, deno_list, save_dir="denoising_results"):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    rows = len(clean_list)
    cols = 3
    titles = ["Clean", "Blur+Noise", "Denoised"]

    fig_w = 3.5  # ~IEEE single-column width (inches)
    per_row_h = 1.0  # height per image row (tweak smaller to squeeze more)
    fig_h = max(0.8, per_row_h * rows)  # total figure height

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(fig_w, fig_h),
        gridspec_kw=dict(wspace=0.0, hspace=0.1)  # kill inter-subplot gaps
    )
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    # remove outer page margins too
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)

    for r in range(rows):
        clean = _to_np_img(clean_list[r])
        noisy = _to_np_img(noisy_list[r])
        deno = _to_np_img(deno_list[r])

        panels = [clean, noisy, deno]

        # prepare (H,W,1) versions for metrics
        clean3 = _to_hw3(clean)
        noisy3 = _to_hw3(noisy)
        deno3 = _to_hw3(deno)

        # metrics vs clean (skip for the clean column)
        metrics = [
            None,
            (compute_psnr(clean3, noisy3),    compute_ssim(clean3, noisy3)),
            (compute_psnr(clean3, deno3),     compute_ssim(clean3, deno3)),
        ]

        for c in range(cols):
            ax = axes[r, c]
            ax.imshow(panels[c], cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            ax.axis("off")

            # small PSNR/SSIM label (skip for the clean column)
            if c > 0:
                psnr, ssim = metrics[c]
                label = f"PSNR {psnr:.2f}dB\nSSIM {ssim:.3f}"
                ax.text(
                    0.02, 0.98, label,
                    transform=ax.transAxes,
                    fontsize=6, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
                )

    out_path = "blur_denoise_grid.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved grid to: {out_path}")
