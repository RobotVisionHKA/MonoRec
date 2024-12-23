import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils import to, PLYSaver, DS_Wrapper
import matplotlib.pyplot as plt

import torch.nn.functional as F
import open3d as o3d
import os
import shutil
import numpy as np

def main(config):
    logger = config.get_logger('test')

    output_dir = Path(config.config.get("output_dir", "saved"))
    output_dir.mkdir(exist_ok=True, parents=True)
    depth_dir = output_dir / 'depth_maps'
    img_dir = output_dir / 'imgs'

    if os.path.isdir(depth_dir):
        shutil.rmtree(depth_dir)
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir)
    os.mkdir(depth_dir)
    os.mkdir(img_dir)

    file_name = config.config.get("file_name", "pc.ply")
    use_mask = config.config.get("use_mask", True)
    roi = config.config.get("roi", None)

    max_d = config.config.get("max_d", 30)
    min_d = config.config.get("min_d", 1.0)

    start = config.config.get("start", 0)
    end = config.config.get("end", -1)

    # setup data_loader instances
    data_loader = DataLoader(DS_Wrapper(config.initialize('data_set', module_data), start=start, end=end), batch_size=1, shuffle=False, num_workers=8)

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    pcd_combined = o3d.geometry.PointCloud()

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data = to(data, device)

            result = model(data)
            if not isinstance(result, dict):
                result = {"result": result[0]}
            output = result["result"]
            
            pose = data["keyframe_pose"][0].cpu().detach().numpy()
            intrinsic = data["keyframe_intrinsics"][0].cpu().detach().numpy()
            img = data["keyframe"][0].permute(1, 2, 0).cpu().detach().numpy()
            depth = output[0, 0].cpu().detach().numpy()
            
            plt.imsave(f"{depth_dir}/{i}.png", depth.squeeze())
            plt.imsave(f"{img_dir}/{i}.png", img + 0.5)

            img = np.array((img + 0.5)*255)
            img = img.astype(np.uint8)
            img = img.copy()

            msk = (min_d <= depth) & (depth <= max_d)
            depth[msk] = 0.0
            color_raw = o3d.geometry.Image(img)
            depth_raw = o3d.geometry.Image(1/depth)
            rgbd = o3d.geometry.create_rgbd_image_from_color_and_depth(color_raw, depth_raw, depth_scale = 1.0, depth_trunc = max_d)
            
            intrinsic_raw = o3d.camera.PinholeCameraIntrinsic(
                width=img.shape[1],
                height=img.shape[0],
                fx=intrinsic[0,0],
                fy=intrinsic[1,1],
                cx=intrinsic[0,2],
                cy=intrinsic[1,2]
            )
            pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, intrinsic_raw, np.linalg.inv(pose))
            pcd_combined+=pcd

    pcd_combined.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud(str(output_dir / file_name), pcd_combined, write_ascii=True, compressed=False)
    o3d.visualization.draw_geometries([pcd_combined])

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
