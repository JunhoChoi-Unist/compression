import pathlib
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from codec.raft.model import RAFT
from dataset import InterTSDFDataset

warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def surface_aware_loss(sdf_pred, sdf_gt, threshold=0.5):
    """
    Compute loss that focuses on surface regions
    """
    surface_mask = (torch.abs(sdf_gt) < threshold).float()

    surface_loss = F.mse_loss(sdf_pred * surface_mask, sdf_gt * surface_mask)

    # Add gradient loss for surface consistency
    grad_pred = torch.gradient(sdf_pred, dim=[2, 3, 4])
    grad_gt = torch.gradient(sdf_gt, dim=[2, 3, 4])
    grad_loss = sum(
        [
            F.mse_loss(gp * surface_mask, gg * surface_mask)
            for gp, gg in zip(grad_pred, grad_gt)
        ]
    )

    return surface_loss + 0.1 * grad_loss


def flow_regularization_loss(flow):
    """
    Regularization for 3D scene flow
    """
    # Smoothness loss - neighboring voxels should have similar flow
    grad_x = torch.diff(flow, dim=2)
    grad_y = torch.diff(flow, dim=3)
    grad_z = torch.diff(flow, dim=4)

    smoothness_loss = (grad_x**2).mean() + (grad_y**2).mean() + (grad_z**2).mean()

    # Sparsity loss - encourage sparse flow
    sparsity_loss = torch.abs(flow).mean()

    return smoothness_loss + 0.01 * sparsity_loss


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    # else "mps"    # disable due to no support on deconv
    # if torch.backends.mps.is_available()
    else "cpu"
)
EPOCHS = 100
BATCH_SIZE = 1
SAVE_DIR = pathlib.Path("checkpoints/intercodec")

if __name__ == "__main__":
    dataset = InterTSDFDataset(
        dataset="MPEG", scene="longdress_voxelized", mode="train"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

    model = RAFT().to(DEVICE)
    best_loss = float("inf")
    continue_epoch = 0

    checkpoint_path = "checkpoints/intercodec/raft_ep072.pth"
    if pathlib.Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        # best_loss = checkpoint["loss"]
        # continue_epoch = checkpoint["epoch"] + 1
        print(
            f"Resuming training from epoch {continue_epoch} with loss {best_loss:.3e}"
        )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model = torch.nn.DataParallel(model)
    for epoch in range(continue_epoch, EPOCHS):
        epoch_loss = 0.0
        epoch_distortion_loss = 0.0
        epoch_regularization_loss = 0.0
        flow_init = None
        model.train()
        for batch_idx, (sdf_blocks, t) in enumerate(tqdm(dataloader, total=1088)):
            sdf_blocks0, sdf_blocksB, sdf_blocks1 = sdf_blocks
            sdf_blocks0 = sdf_blocks0.to(DEVICE)
            sdf_blocksB = sdf_blocksB.to(DEVICE)
            sdf_blocks1 = sdf_blocks1.to(DEVICE)
            N, _, D, H, W = sdf_blocks0.shape
            t = t.view(-1, 1, 1, 1, 1).to(DEVICE)
            optimizer.zero_grad()
            flow_predictions = model(
                sdf_blocks1,
                sdf_blocks0,
                t,
                iters=12,
                flow_init=flow_init,
                upsample=True,
                test_mode=False,
            )
            distortion_loss = 0.0
            regularization_loss = 0.0
            for i, flow in enumerate(flow_predictions):
                _lmbda = 0.8 ** (len(flow_predictions) - i - 1)
                # flow = model.interpolate_flow(flow, t)
                sdf_hat = model.module.warp(sdf_blocks0, flow)
                distortion_loss += _lmbda * surface_aware_loss(sdf_hat, sdf_blocksB)
                regularization_loss += _lmbda * flow_regularization_loss(flow)
            distortion_loss = distortion_loss / sum(
                0.8**i for i in range(len(flow_predictions))
            )
            regularization_loss = regularization_loss / sum(
                0.8**i for i in range(len(flow_predictions))
            )
            loss = distortion_loss + 0.1 * regularization_loss
            epoch_loss += loss
            epoch_distortion_loss += distortion_loss
            epoch_regularization_loss += regularization_loss
            loss.backward()
            optimizer.step()

        epoch_loss /= batch_idx
        epoch_distortion_loss /= batch_idx
        epoch_regularization_loss /= batch_idx

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = (
                SAVE_DIR / f"{model.module.__class__.__name__}_ep{epoch:03d}.pth"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "distortion": epoch_distortion_loss,
                },
                save_path,
            )
            print(
                f"Epoch {epoch}: distortion={epoch_distortion_loss:.3e} reg={epoch_regularization_loss:.3e} [SAVED]"
            )
        else:
            print(
                f"Epoch {epoch}: distortion={epoch_distortion_loss:.3e} reg={epoch_regularization_loss:.3e}"
            )
