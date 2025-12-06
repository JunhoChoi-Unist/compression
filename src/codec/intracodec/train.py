import math
import pathlib

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from codec.intracodec.model import HyperPrior
from dataset import IntraTSDFDataset

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    # else "mps"    # disable due to no support on deconv
    # if torch.backends.mps.is_available()
    else "cpu"
)
EPOCHS = 1000
BATCH_SIZE = 6
SAVE_DIR = pathlib.Path("checkpoints/intracodec")

if __name__ == "__main__":
    dataset = IntraTSDFDataset(
        dataset="MPEG", scene="longdress_voxelized", mode="train"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    model = HyperPrior(N=64, M=32).to(DEVICE)
    best_loss = float("inf")
    continue_epoch = 0

    # Use list of tuples instead of dict to be able to later check the elements are unique and there is no intersection
    parameters = [
        p for n, p in model.named_parameters() if not n.endswith(".quantiles")
    ]
    aux_parameters = [
        p for n, p in model.named_parameters() if n.endswith(".quantiles")
    ]

    # # Make sure we don't have an intersection of parameters
    # parameters_name_set = set(n for n,p in parameters)
    # aux_parameters_name_set = set(n for n, p in aux_parameters)
    # assert len(parameters) == len(parameters_name_set)
    # assert len(aux_parameters) == len(aux_parameters_name_set)

    # inter_params = parameters_name_set & aux_parameters_name_set
    # union_params = parameters_name_set | aux_parameters_name_set

    # assert len(inter_params) == 0
    # assert len(union_params) - len(dict(net.named_parameters()).keys()) == 0

    optimizer = optim.Adam(parameters, lr=1e-2)  # 1e-3: exploded
    # ~800epochs with 1e-3 lr: D:1.5e-05 S:0.035 --> 1e-4
    aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)

    checkpoint_path = "None"
    # checkpoint_path = "checkpoints/intracodec/HyperPrior_ep756.pth"
    # "level10: Epoch 756: distortion=7.057e-05, rate=0.021375, sign=0.003229 [SAVED]"
    if pathlib.Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model = HyperPrior.from_state_dict(checkpoint["model_state_dict"]).to(DEVICE)
        best_loss = checkpoint["loss"]
        continue_epoch = checkpoint["epoch"] + 1
        parameters = [
            p for n, p in model.named_parameters() if not n.endswith(".quantiles")
        ]
        aux_parameters = [
            p for n, p in model.named_parameters() if n.endswith(".quantiles")
        ]
        optimizer = optim.Adam(parameters, lr=1e-2)  # 1e-3: exploded
        aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer_state_dict"])
        print(
            f"Resuming training from epoch {continue_epoch} with loss {best_loss:.3e}"
        )

    for epoch in range(continue_epoch, EPOCHS):
        epoch_loss = 0.0
        epoch_distortion_loss = 0.0
        epoch_rate_loss = 0.0
        epoch_sign_loss = 0.0
        model.train()
        for batch_idx, sdf_blocks in enumerate(tqdm(dataloader, total=1088)):
            x = sdf_blocks.to(DEVICE)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            out = model(x)
            magn_hat = out["magn_hat"]
            sign_hat = out["sign_hat"]
            y_likelihoods = out["likelihoods"]["y"]
            z_likelihoods = out["likelihoods"]["z"]
            # x_hat = torch.abs(magn_hat) * torch.sign(x)
            threshold = 0.5  # threshold 0.1 --> bumpy surface
            mask = torch.abs(x) < threshold
            distortion_loss = F.mse_loss(
                torch.abs(magn_hat) * mask,
                torch.abs(x) * mask,
            )
            N, _, D, H, W = x.size()
            num_voxels = N * D * H * W
            rate_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_voxels)
            rate_loss += torch.log(z_likelihoods).sum() / (-math.log(2) * num_voxels)
            is_positive = (x >= 0).squeeze(1).long()
            sign_rate = F.cross_entropy(sign_hat, is_positive, reduction="sum") / (
                math.log(2) * num_voxels
            )
            rp = 7
            mu = rp * math.log10(200000) / 11
            lmbda = 1 / (10**mu)
            loss = distortion_loss + lmbda * (rate_loss + sign_rate)
            epoch_loss += loss
            epoch_distortion_loss += distortion_loss
            epoch_rate_loss += rate_loss
            epoch_sign_loss += sign_rate
            loss.backward()
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

        epoch_loss /= batch_idx
        epoch_distortion_loss /= batch_idx
        epoch_rate_loss /= batch_idx
        epoch_sign_loss /= batch_idx

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = SAVE_DIR / f"{model.__class__.__name__}_ep{epoch:03d}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "distortion": epoch_distortion_loss,
                    "rate": epoch_rate_loss,
                    "sign": epoch_sign_loss,
                },
                save_path,
            )

            print(
                f"Epoch {epoch}: distortion={epoch_distortion_loss:.3e}, rate={epoch_rate_loss:.6f}, sign={epoch_sign_loss:.6f} [SAVED]"
            )
        else:
            print(
                f"Epoch {epoch}: distortion={epoch_distortion_loss:.3e}, rate={epoch_rate_loss:.6f}, sign={epoch_sign_loss:.6f}"
            )
