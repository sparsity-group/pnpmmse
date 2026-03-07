import torch
import time
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Local Imports
from utils import load_dataset
from lpn_mnist import LPN

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 500
DATALOADER_NUM_WORKERS = 8
VALIDATE_EVERY_N_STEPS = 5000
SAVE_EVERY_N_STEPS = 2_000
NOISE_SIGMA = .2
BETA = 10
IN_DIM = 1
HIDDEN = 64
NUM_STEPS = 40_000
SAVE_EVERY = 1000
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(int(time.time()))
np.random.seed(int(time.time()))
if not torch.cuda.is_available():
    raise Exception("CUDA is not available")

# Model, optimizer, summary log
model = LPN(in_dim=IN_DIM, hidden=HIDDEN, beta=BETA).to("cuda")
model.init_weights(-6, 0.1)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=3e-5)
writer = SummaryWriter(log_dir="training_log")

train_dataloader = torch.utils.data.DataLoader(
    load_dataset("train"),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=DATALOADER_NUM_WORKERS,
)

global_step = 0
progress_bar = tqdm(total=NUM_STEPS, desc="Train")


while True:
    for batch in train_dataloader:
        model.train()
        clean = batch["image"].to("cuda")
        noise = torch.randn_like(clean)
        noisy_images = clean + NOISE_SIGMA * noise
        out = model(noisy_images)
        loss = F.mse_loss(out, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if global_step % SAVE_EVERY == 0:
            torch.save({
                "iteration": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item(),
            }, "model.pt")

        global_step += 1
        if global_step >= NUM_STEPS:
            break
    if global_step >= NUM_STEPS:
        break
