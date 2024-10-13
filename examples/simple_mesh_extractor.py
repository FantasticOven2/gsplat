
import tyro
import torch

from examples.simple_trainer_2dgs import Runner, Config
from utils import MeshExtractor

DEVICE = "cuda"

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Mesh extractor parameters")
    parser.add_argument("--ckpt", type=str, help="path to gaussian splatting checkpoint")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF")
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args()

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    runner = Runner(cfg)

    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt["splats"][k]
    runner.eval(step=ckpt["step"])
    runner.render_traj(step=ckpt["step"])

    mesh_extractor = MeshExtractor(
        runner=runner,
        voxel_size=args.voxel_size,
        depth_trunc=args.depth_trunc,
        sdf_trunc=args.sdf_trunc,
        num_cluster=args.num_cluster,
        mesh_res=args.mesh_res,
    )

    # Step 1: render train views RGB/Depth
    mesh_extractor.reconstruction()

    # Step 2: extract mesh
    depth_trunc = (mesh_extractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
    voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
    sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    # Step 3: save mesh to file
    o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
    print(f"mesh saved at {os.path.join(train_dir, name)}")

    # Step 4: Post-process the mesh and save, saving the largest N_clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace(".ply", "_post.ply")), mesh_post)
    print(f"mesh post processed saved at {os.path.join(train_dir, name.replace(".ply", "_post.ply"))}")