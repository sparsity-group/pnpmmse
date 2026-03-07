from utils.utils import load_dataset
from networks.lpn_mnist import LPN
import torch
import matplotlib.pyplot as plt

DEVICE = "cuda"
NOISE_SIGMA = .2
NUM_ITERATIONS = 50
BLUR_SIGMA = 1
BLUR_KERNEL_SIZE = 3
MODEL_PATH = "model/model.pt"
LPN_HIDDEN_LAYERS = 64
LPN_BETA = 10 # Softplus activation
LPN_NUM_CHANNELS = 1
NUMBER_TEST_IMAGES = 1


def main():
    model = LPN(in_dim=LPN_NUM_CHANNELS, hidden=LPN_HIDDEN_LAYERS, beta=LPN_BETA).to(DEVICE)
    test_ds = load_dataset("test")
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    # model = ProxL2(mu=1)
    torch.set_grad_enabled(False)
    item = test_ds[0]
    clean = item["image"].unsqueeze(0).to(DEVICE)
    noisy = clean + NOISE_SIGMA * torch.randn_like(clean)
    iter = noisy
    for i in range(0,5):
        iter = model(noisy - iter)
    img = iter.detach().cpu().squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.savefig("test.png")

if __name__ == "__main__":
    main()