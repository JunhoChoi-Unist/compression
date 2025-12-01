import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

SDF_ROOT = pathlib.Path("data/SDF/5mm")


class IntraTSDFDataset(IterableDataset):
    def __init__(
        self, dataset: str = "MPEG", scene: str = "longdress_voxelized", mode="train"
    ):
        super().__init__()
        self.mode = mode
        self.dataset = dataset
        self.scene = scene
        self.npzfiles = [
            npzfile
            for npzfile in (SDF_ROOT / dataset / scene).glob("*.npz")
            if npzfile.name != "common.npz"
        ]
        if mode == "test":
            self.npzfiles = sorted(self.npzfiles)
        else:
            random.shuffle(self.npzfiles)

        with np.load(SDF_ROOT / dataset / scene / "common.npz") as common:
            self.voxel_size = common["voxel_size"].astype(np.float32)
        self.sdf_trunc = np.linalg.norm(self.voxel_size)

    def _blockify(self, sdf, block_size=64):
        # This should return an iterable of blocks
        D, H, W = sdf.shape

        # Pad if necessary
        pad_d = (block_size - D % block_size) % block_size
        pad_h = (block_size - H % block_size) % block_size
        pad_w = (block_size - W % block_size) % block_size

        sdf = np.pad(
            sdf,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=-1.0,
        )

        D, H, W = sdf.shape

        blocks = sdf.reshape(
            D // block_size,
            block_size,
            H // block_size,
            block_size,
            W // block_size,
            block_size,
        ).transpose(0, 2, 4, 1, 3, 5)
        return blocks

    def __iter__(self):
        # This generator will be called by the DataLoader
        for npzfile in self.npzfiles:
            npzdata = np.load(npzfile)
            sdf = npzdata["sdf"].astype(np.float32)
            sdf = (sdf / self.sdf_trunc).clip(-1.0, 1.0)

            blocks = self._blockify(sdf)
            nD, nH, nW, block_sizeD, block_sizeH, block_sizeW = blocks.shape
            if self.mode == "test":
                blocks = blocks.reshape(
                    1, nD, nH, nW, 1, block_sizeD, block_sizeH, block_sizeW
                )
                min_bound = npzdata["min_bound"].astype(np.float32)
                for block in blocks:
                    yield torch.from_numpy(block), min_bound, npzfile.stem
            else:
                blocks = blocks.reshape(-1, 1, block_sizeD, block_sizeH, block_sizeW)
                for block in blocks:
                    # check if the block contains any surface voxels
                    if block.max() * block.min() <= 0:
                        yield torch.from_numpy(block)
                    else:
                        continue


if __name__ == "__main__":
    # set batch size to 1 for testing
    dataset = IntraTSDFDataset(mode="test")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    for batch in dataloader:
        block, min_bound, filename = batch
        print(block.shape)
    print("Done")
