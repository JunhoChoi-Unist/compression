import pathlib
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from codec.intercodec.model import RAFT
from dataset import InterTSDFDataset

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    # else "mps"    # disable due to no support on deconv
    # if torch.backends.mps.is_available()
    else "cpu"
)
EPOCHS = 100
BATCH_SIZE = 3
SAVE_DIR = pathlib.Path("checkpoints/intercodec")

if __name__ == "__main__":
    dataset = InterTSDFDataset(
        dataset="MPEG", scene="longdress_voxelized", mode="train"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    model = RAFT().to(DEVICE)
    best_loss = float("inf")
    continue_epoch = 0

    checkpoint_path = "checkpoints/intercodec/raft_ep020.pth"
    if pathlib.Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_loss = checkpoint["loss"]
        continue_epoch = checkpoint["epoch"] + 1
        print(
            f"Resuming training from epoch {continue_epoch} with loss {best_loss:.3e}"
        )

    # parameters = set(
    #     p for n, p in model.named_parameters() if not n.endswith(".quantiles")
    # )
    # aux_parameters = set(
    #     p for n, p in model.named_parameters() if n.endswith(".quantiles")
    # )
    # optimizer = optim.Adam(parameters, lr=1e-5)
    # aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    for epoch in range(continue_epoch, EPOCHS):
        epoch_loss = 0.0
        epoch_distortion_loss = 0.0
        flow_init = None
        for batch_idx, (sdf_blocks, t) in enumerate(tqdm(dataloader, total=1088)):
            sdf_blocks0, sdf_blocksB, sdf_blocks1 = sdf_blocks
            sdf_blocks0 = sdf_blocks0.to(DEVICE)
            sdf_blocksB = sdf_blocksB.to(DEVICE)
            sdf_blocks1 = sdf_blocks1.to(DEVICE)
            t = t.view(-1, 1, 1, 1, 1).to(DEVICE)
            optimizer.zero_grad()
            flow_predictions = model(
                sdf_blocks1,
                sdf_blocks0,
                iters=12,
                flow_init=flow_init,
                upsample=True,
                test_mode=False,
            )
            flow_predictions = [flow * t for flow in flow_predictions]
            distortion_loss = 0.0
            for i, flow in enumerate(flow_predictions):
                _lmbda = 0.8 ** (len(flow_predictions) - i - 1)
                sdf_hat = model.warp(sdf_blocks0, flow)
                distortion_loss += _lmbda * F.mse_loss(sdf_hat, sdf_blocksB)
            distortion_loss = distortion_loss / sum(
                0.8**i for i in range(len(flow_predictions))
            )
            loss = distortion_loss
            epoch_loss += loss
            epoch_distortion_loss += distortion_loss
            loss.backward()
            optimizer.step()

            # aux_loss = model.aux_loss()
            # aux_loss.backward()
            # aux_optimizer.step()

        epoch_loss /= batch_idx
        epoch_distortion_loss /= batch_idx
        print(f"Epoch {epoch}: distortion={epoch_distortion_loss:.3e}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = SAVE_DIR / f"{model.__class__.__name__}_ep{epoch:03d}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "distortion": epoch_distortion_loss,
                },
                save_path,
            )
