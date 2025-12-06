import math
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

SDF_ROOT = pathlib.Path("data/SDF/5mm")


class InterTSDFDataset(IterableDataset):
    def __init__(
        self,
        dataset: str = "MPEG",
        scene: str = "longdress_voxelized",
        gop=4,
        mode="train",
    ):
        super().__init__()
        self.gop = gop
        self.mode = mode
        self.dataset = dataset
        self.scene = scene
        self.npzfiles = [
            npzfile
            for npzfile in (SDF_ROOT / dataset / scene).glob("*.npz")
            if npzfile.name != "common.npz"
        ]
        self.npzfiles = sorted(self.npzfiles)
        if mode == "test":
            self.npzfiles = self.npzfiles[:28]
        else:
            pass

        with np.load(SDF_ROOT / dataset / scene / "common.npz") as common:
            self.voxel_size = common["voxel_size"].astype(np.float32)
        self.sdf_trunc = np.linalg.norm(self.voxel_size)

    def __iter__(self):
        # This generator will be called by the DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            npzfiles = self.npzfiles
        else:
            per_worker = int(
                math.ceil(len(self.npzfiles) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.npzfiles))
            npzfiles = self.npzfiles[iter_start:iter_end]
        npzfile0 = npzfiles.pop(0)
        npzdata0 = np.load(npzfile0)
        sdf0 = npzdata0["sdf"].astype(np.float32)
        sdf0 = (sdf0 / self.sdf_trunc).clip(-1.0, 1.0)
        while len(npzfiles) >= self.gop - 1:
            npzfile1 = npzfiles.pop(self.gop - 1 - 1)
            npzdata1 = np.load(npzfile1)
            sdf1 = npzdata1["sdf"].astype(np.float32)
            sdf1 = (sdf1 / self.sdf_trunc).clip(-1.0, 1.0)
            for i in range(self.gop - 2):
                npzfileB = npzfiles.pop(0)
                npzdataB = np.load(npzfileB)
                sdfB = npzdataB["sdf"].astype(np.float32)
                sdfB = (sdfB / self.sdf_trunc).clip(-1.0, 1.0)
                t = torch.Tensor([(i + 1) / (self.gop - 1)])
                d0, h0, w0 = sdf0.shape
                dB, hB, wB = sdfB.shape
                d1, h1, w1 = sdf1.shape

                max_d = max(d0, dB, d1)
                max_h = max(h0, hB, h1)
                max_w = max(w0, wB, w1)

                pad_d0 = max_d - d0
                pad_h0 = max_h - h0
                pad_w0 = max_w - w0
                _sdf0 = np.pad(
                    sdf0,
                    ((0, pad_d0), (0, pad_h0), (0, pad_w0)),
                    mode="constant",
                    constant_values=-1.0,
                )

                pad_dB = max_d - dB
                pad_hB = max_h - hB
                pad_wB = max_w - wB
                _sdfB = np.pad(
                    sdfB,
                    ((0, pad_dB), (0, pad_hB), (0, pad_wB)),
                    mode="constant",
                    constant_values=-1.0,
                )

                pad_d1 = max_d - d1
                pad_h1 = max_h - h1
                pad_w1 = max_w - w1
                _sdf1 = np.pad(
                    sdf1,
                    ((0, pad_d1), (0, pad_h1), (0, pad_w1)),
                    mode="constant",
                    constant_values=-1.0,
                )
                blocks0 = blockify(_sdf0, block_size=128)
                blocksB = blockify(_sdfB, block_size=128)
                blocks1 = blockify(_sdf1, block_size=128)
                nD, nH, nW, block_sizeD, block_sizeH, block_sizeW = blocks0.shape
                if self.mode == "test":
                    blocks0 = blocks0.reshape(
                        1, nD, nH, nW, 1, block_sizeD, block_sizeH, block_sizeW
                    )
                    blocksB = blocksB.reshape(
                        1, nD, nH, nW, 1, block_sizeD, block_sizeH, block_sizeW
                    )
                    blocks1 = blocks1.reshape(
                        1, nD, nH, nW, 1, block_sizeD, block_sizeH, block_sizeW
                    )
                    min_bound0 = npzdata0["min_bound"].astype(np.float32)
                    min_boundB = npzdataB["min_bound"].astype(np.float32)
                    min_bound1 = npzdata1["min_bound"].astype(np.float32)
                    for block0, blockB, block1 in zip(blocks0, blocksB, blocks1):
                        yield (
                            (
                                torch.from_numpy(block0),
                                torch.from_numpy(blockB),
                                torch.from_numpy(block1),
                            ),
                            (t),
                            (min_bound0, min_boundB, min_bound1),
                            (npzfile0.stem, npzfileB.stem, npzfile1.stem),
                        )
                else:
                    blocks0 = blocks0.reshape(
                        -1, 1, block_sizeD, block_sizeH, block_sizeW
                    )
                    blocksB = blocksB.reshape(
                        -1, 1, block_sizeD, block_sizeH, block_sizeW
                    )
                    blocks1 = blocks1.reshape(
                        -1, 1, block_sizeD, block_sizeH, block_sizeW
                    )
                    for block0, blockB, block1 in zip(blocks0, blocksB, blocks1):
                        # check if the block contains any surface voxels
                        if (
                            block0.max() * block0.min() <= 0
                            or blockB.max() * blockB.min() <= 0
                            or block1.max() * block1.min() <= 0
                        ):
                            yield (
                                (
                                    torch.from_numpy(block0),
                                    torch.from_numpy(blockB),
                                    torch.from_numpy(block1),
                                ),
                                (t),
                            )
                        else:
                            continue
            npzfile0 = npzfile1
            npzdata0 = npzdata1
            sdf0 = sdf1
            blocks0 = blocks1


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
        self.npzfiles = sorted(self.npzfiles)
        if mode == "test":
            pass
        else:
            # self.npzfiles = self.npzfiles[:4]
            random.shuffle(self.npzfiles)

        with np.load(SDF_ROOT / dataset / scene / "common.npz") as common:
            self.voxel_size = common["voxel_size"].astype(np.float32)
        self.sdf_trunc = np.linalg.norm(self.voxel_size)

    def __iter__(self):
        # This generator will be called by the DataLoader
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            npzfiles = self.npzfiles
        else:
            per_worker = int(
                math.ceil(len(self.npzfiles) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.npzfiles))
            npzfiles = self.npzfiles[iter_start:iter_end]
        for npzfile in npzfiles:
            npzdata = np.load(npzfile)
            sdf = npzdata["sdf"].astype(np.float32)
            sdf = (sdf / self.sdf_trunc).clip(-1.0, 1.0)

            blocks = blockify(sdf, block_size=128)
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


def blockify(sdf, block_size=128):
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


if __name__ == "__main__":
    # set batch size to 1 for testing
    dataset = IntraTSDFDataset(mode="test")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    for batch in dataloader:
        block, min_bound, filename = batch
        print(block.shape)
    print("Done")
