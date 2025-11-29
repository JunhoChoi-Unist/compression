import pathlib

import numpy as np
import torch
import trimesh
from skimage import measure
from torch.utils.data import DataLoader

from codec.intracodec.model import HyperPrior
from dataset import IntraTSDFDataset

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    # else "mps"  # disable due to no support on deconv
    # if torch.backends.mps.is_available()
    else "cpu"
)
SAVE_DIR = pathlib.Path("results/intracodec")

if __name__ == "__main__":
    dataset = IntraTSDFDataset(dataset="MPEG", scene="longdress_voxelized", mode="test")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    model = HyperPrior().to(DEVICE)
    checkpoint_path = "checkpoints/intracodec/HyperPrior_ep029.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    model = HyperPrior.from_state_dict(state_dict).to(DEVICE)
    model.update()
    model.eval()
    # model = torch.compile(model)
    for batch_idx, (sdf_blocks, min_bound, filename) in enumerate(dataloader):
        _, nD, nH, nW, _, block_sizeD, block_sizeH, block_sizeW = sdf_blocks.shape
        sdf_hat = -torch.ones(
            (block_sizeD * nD, block_sizeH * nH, block_sizeW * nW)
        ).float()

        for i in range(nD):
            for j in range(nH):
                for k in range(nW):
                    x = sdf_blocks[:1, i, j, k].to(DEVICE)
                    with torch.no_grad():
                        out = model.compress(x)
                        strings = out["strings"]
                        shape = out["shape"]
                        x_hat = model.decompress(strings, shape)["x_hat"] * torch.sign(
                            x
                        )
                    sdf_hat[
                        i * block_sizeD : (i + 1) * block_sizeD,
                        j * block_sizeH : (j + 1) * block_sizeH,
                        k * block_sizeW : (k + 1) * block_sizeW,
                    ] = x_hat[0, 0].cpu()
        sdf_hat = sdf_hat.numpy()

        v, f, n, _ = measure.marching_cubes(sdf_hat, level=0.0)
        f = np.flip(f, axis=1)
        v = v * dataset.voxel_size + min_bound[0].numpy()
        mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=True)
        objfile = SAVE_DIR / dataset.dataset / dataset.scene / f"{filename[0]}.obj"
        objfile.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(objfile)
        print(f"Saved {objfile}")

    print("Done")
