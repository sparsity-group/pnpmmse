import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from lpn_512 import LPN
from utils import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
EXP_DIR          = "./results"
DATA_ROOT        = "./mayoct"
SIGMA_MIN        = 0.01
SIGMA_MAX        = 0.2
TRAIN_BATCH_SIZE = 64
NUM_STEPS        = 40000
LR               = 1e-4
LR_MIN           = 1e-6
SAVE_EVERY       = 1000
SEED             = 0
RESUME_FROM      = None
IN_DIM           = 1
BETA             = 10
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(EXP_DIR, exist_ok=True)

model = LPN(in_dim=IN_DIM, beta=BETA).to(device)
if RESUME_FROM:
    checkpoint = torch.load(RESUME_FROM)
    model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEPS, eta_min=LR_MIN)
loss_func = torch.nn.MSELoss()
writer = SummaryWriter(log_dir=f"{EXP_DIR}/tb")

train_dataloader = torch.utils.data.DataLoader(
    load_dataset(DATA_ROOT, "train"),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4,
)


global_step = 0
best_val_loss = float("inf")
progress_bar = tqdm(total=NUM_STEPS, desc="Train")

while True:
    for batch in train_dataloader:
        model.train()
        clean = batch["image"].to(device)
        sigma = torch.empty(clean.shape[0], 1, 1, 1, device=device).uniform_(SIGMA_MIN, SIGMA_MAX)
        noisy = clean + sigma * torch.randn_like(clean)
        out = model(noisy)
        loss = loss_func(out, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if global_step % SAVE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(train_dataloader))
                clean_val = val_batch["image"].to(device)
                sigma_val = torch.empty(clean_val.shape[0], 1, 1, 1, device=device).uniform_(SIGMA_MIN, SIGMA_MAX)
                noisy_val = clean_val + sigma_val * torch.randn_like(clean_val)
                val_loss = loss_func(model(noisy_val), clean_val).item()
            writer.add_scalar("Loss/val", val_loss, global_step)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", val_loss=f"{val_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # always save latest
            torch.save({
                "iteration": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss.item(),
                "val_loss": val_loss,
            }, f"{EXP_DIR}/model.pt")

            # save best separately
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "iteration": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                    "val_loss": val_loss,
                }, f"{EXP_DIR}/best_model.pt")

        global_step += 1
        if global_step >= NUM_STEPS:
            break
    if global_step >= NUM_STEPS:
        break

progress_bar.close()
writer.close()

# final save
torch.save({
    "iteration": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss.item(),
    "val_loss": val_loss,
}, f"{EXP_DIR}/model.pt")