import pathlib
from math import ceil

import numpy as np
import open3d as o3d
import trimesh
from pysdf import SDF
from skimage import measure
from tqdm import tqdm

ORIGINAL_ROOT = pathlib.Path("data/Original")
SDF_ROOT = pathlib.Path("data/SDF") / "5mm"


def get_voxel_size(dataset: str = "MPEG", scene: str = "longdress_voxelized"):
    common_path = SDF_ROOT / dataset / scene / "common.npz"
    if not common_path.exists():
        objfiles = list((ORIGINAL_ROOT / dataset / scene).glob("*.obj"))
        largest_extents = np.zeros((3,), dtype=np.float32)
        for objfile in tqdm(objfiles, desc=f"(calc. voxel size) {dataset}/{scene}"):
            mesh = trimesh.load_mesh(objfile, process=True)
            largest_extents = np.maximum(largest_extents, mesh.extents)
        largest_extent = np.max(largest_extents)
        voxel_length = largest_extent / 400.0  # 1.8m / 400 ~= 5mm
        voxel_size = np.array(
            [voxel_length, voxel_length, voxel_length], dtype=np.float32
        )
        common_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(common_path, voxel_size=voxel_size)

    common = np.load(common_path)
    voxel_size = common["voxel_size"].astype(np.float32)
    return voxel_size


def convert_mesh(dataset: str = "MPEG", scene: str = "longdress_voxelized"):
    voxel_size = get_voxel_size(dataset, scene)
    objfiles = list((ORIGINAL_ROOT / dataset / scene).glob("*.obj"))
    (SDF_ROOT / dataset / scene).mkdir(parents=True, exist_ok=True)
    for objfile in tqdm(objfiles, desc=f"(calc. sdf) {dataset}/{scene}"):
        npz_path = SDF_ROOT / dataset / scene / f"{objfile.stem}.npz"
        # if npz_path.exists():
        #     continue
        mesh = trimesh.load_mesh(objfile, process=True)
        min_bound = mesh.bounds[0]

        # query points
        x = (
            np.arange(0, ceil(mesh.extents[0] / voxel_size[0]) + 1) * voxel_size[0]
            + min_bound[0]
        )
        y = (
            np.arange(0, ceil(mesh.extents[1] / voxel_size[1]) + 1) * voxel_size[1]
            + min_bound[1]
        )
        z = (
            np.arange(0, ceil(mesh.extents[2] / voxel_size[2]) + 1) * voxel_size[2]
            + min_bound[2]
        )
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        query_points = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
        resolution = (len(x), len(y), len(z))

        # compute sign (pysdf)
        f = SDF(mesh.vertices, mesh.faces)
        signed_distance = f(query_points.reshape((-1, 3))).reshape(resolution)
        sign = np.sign(signed_distance)

        # compute unsigned distance (open3d)
        o3d_scene = o3d.t.geometry.RaycastingScene()
        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(mesh.vertices),
            triangles=o3d.utility.Vector3iVector(mesh.faces),
        )
        o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
        _ = o3d_scene.add_triangles(o3d_mesh)
        unsigned_distance = o3d_scene.compute_distance(query_points)
        unsigned_distance = unsigned_distance.numpy()

        # combine sign and unsigned distance
        sdf = unsigned_distance * sign

        np.savez_compressed(
            npz_path,
            sdf=sdf.astype(np.float32),
            min_bound=min_bound.astype(np.float32),
        )


def convert_sdf(
    dataset: str = "MPEG", scene: str = "longdress_voxelized", truncate=True
):
    voxel_size = get_voxel_size(dataset, scene)
    npzfiles = [
        npzfile
        for npzfile in (SDF_ROOT / dataset / scene).glob("*.npz")
        if not npzfile.name == "common.npz"
    ]
    for npzfile in tqdm(npzfiles, desc=f"(marching cubes) {dataset}/{scene}"):
        objfile = SDF_ROOT / dataset / scene / f"{npzfile.stem}.obj"
        # if objfile.exists():
        #     continue
        npzdata = np.load(npzfile)
        sdf = npzdata["sdf"].astype(np.float32)
        min_bound = npzdata["min_bound"].astype(np.float32)
        if truncate:
            sdf_trunc = np.linalg.norm(voxel_size)
            sdf = (sdf / sdf_trunc).clip(-1.0, 1.0)
        v, f, n, _ = measure.marching_cubes(sdf, level=0.0)
        f = np.flip(f, axis=1)
        v = v * voxel_size + min_bound
        mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=True)
        mesh.export(objfile)


if __name__ == "__main__":
    dataset = "MPEG"
    scene = "longdress_voxelized"
    convert_mesh(dataset, scene)
    convert_sdf(dataset, scene)
