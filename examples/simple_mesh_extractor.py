
import os

import tyro
import torch

import open3d as o3d

from tqdm import tqdm
from argparse import ArgumentParser

from simple_trainer_2dgs import Runner, Config
from gsplat.utils import MeshExtractor

DEVICE = "cuda"



if __name__ == "__main__":
    # Set up command line argument parser
    # parser = ArgumentParser(description="Mesh extractor parameters")
    # parser.add_argument("--ckpt", type=str, help="path to gaussian splatting checkpoint")
    # parser.add_argument("--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF")
    # parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    # parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    # parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    # parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    # args = parser.parse_args()
    voxel_size = -1.0
    depth_trunc = -1.0
    sdf_trunc = -1.0
    num_cluster = 50
    mesh_res = 1024

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    runner = Runner(cfg)

    ckpt = torch.load(cfg.ckpt, map_location=DEVICE)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]

    mesh_extractor = MeshExtractor(
        # voxel_size=args.voxel_size,
        # depth_trunc=args.depth_trunc,
        # sdf_trunc=args.sdf_trunc,
        # num_cluster=args.num_cluster,
        # mesh_res=args.mesh_res,
    )

    # Step 1: render train views RGB/Depth
    data_loader = torch.utils.data.DataLoader(
        runner.trainset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    loader_iter = iter(data_loader)
    pbar = tqdm(range(len(runner.trainset)))

    viewpoint_cam = []
    intrins = []
    render_rgbs = []
    render_depths = []

    with torch.no_grad():
        for step in pbar:
            data = next(loader_iter)
            
            camtoworlds = data["camtoworld"].to(DEVICE) # [1, 4, 4]
            Ks = data["K"].to(DEVICE) # [1, 3, 3]

            height, width = data["image"].shape[1:3]

            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
            ) = runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED"
            )

            render_rgbs.append(colors[..., :3])
            # render_depths.append(render_median)
            render_depths.append(colors[..., 3:])
            viewpoint_cam.append(data["camtoworld"])
            intrins.append(data["K"])

    mesh_extractor.set_viewpoint_stack(
        torch.stack(viewpoint_cam).squeeze(1)
    )
    mesh_extractor.set_Ks(
        torch.stack(intrins).squeeze(1)
    )
    mesh_extractor.set_rgb_maps(
        torch.stack(render_rgbs).squeeze(1)
    )
    mesh_extractor.set_depth_maps(
        torch.stack(render_depths).squeeze(1)
    )

    mesh_extractor.estimate_bounding_sphere()

    # Step 2: extract mesh
    depth_trunc = (mesh_extractor.radius * 2.0) if depth_trunc < 0 else depth_trunc
    voxel_size = (depth_trunc / mesh_res) if voxel_size < 0 else voxel_size
    sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
    mesh = mesh_extractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    import pdb
    pdb.set_trace()

    name = "fused.ply"
    # Step 3: save mesh to file
    o3d.io.write_triangle_mesh(os.path.join(cfg.result_dir, name), mesh)
    print(f"mesh saved at {os.path.join(cfg.result_dir, name)}")

    # Step 4: Post-process the mesh and save, saving the largest N_clusters
    # mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
    # o3d.io.write_triangle_mesh(os.path.join(cfg.result_dir, name.replace(".ply", "_post.ply")), mesh_post)
    # print("mesh post processed saved at {}".format(os.path.join(cfg.result_dir, name.replace('.ply', '_post.ply'))))