# MonoRec
[**Paper**](https://arxiv.org/abs/2011.11814) |  [**Video** (CVPR)](https://youtu.be/XimdlXUamo0) | [**Video** (Reconstruction)](https://youtu.be/-gDSBIm0vgk) | [**Project Page**](https://vision.in.tum.de/research/monorec)

This repository is the official implementation of the paper:

> **MonoRec: Semi-Supervised Dense Reconstruction in Dynamic Environments from a Single Moving Camera**
>
> [Felix Wimbauer*](https://www.linkedin.com/in/felixwimbauer), [Nan Yang*](https://vision.in.tum.de/members/yangn), [Lukas Von Stumberg](https://vision.in.tum.de/members/stumberg), [Niclas Zeller](https://vision.in.tum.de/members/zellern) and [Daniel Cremers](https://vision.in.tum.de/members/cremers)
> 
> [**CVPR 2021** (arXiv)](https://arxiv.org/abs/2011.11814)

<a href="https://youtu.be/-gDSBIm0vgk"><div style="text-align:center"><img src="./pictures/frames.gif" style="height:auto;width:50%"/><img src="./pictures/pointcloud.gif" style="height:auto;width:50%"/></div></a>

If you find our work useful, please consider citing our paper:
```
@InProceedings{wimbauer2020monorec,
  title = {{MonoRec}: Semi-Supervised Dense Reconstruction in Dynamic Environments from a Single Moving Camera},
  author = {Wimbauer, Felix and Yang, Nan and von Stumberg, Lukas and Zeller, Niclas and Cremers, Daniel},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021},
}
```

## 🏗️️ Setup

The `conda` environment for this project can be setup by running the following command:

```shell
conda env create -f environment.yml
```

## 🏃 Running the Example Script

We provide a sample from the KITTI Odometry test set and a script to run MonoRec on it in ``example/``. 
To download the pretrained model and put it into the right place, run ``download_model.sh``. 
You can manually do this by can by downloading the weights from [here](https://vision.in.tum.de/_media/research/monorec/monorec_depth_ref.pth.zip) 
and unpacking the file to ``saved/checkpoints/monorec_depth_ref.pth``.
The example script will plot the keyframe, depth prediction and mask prediction.

```shell
cd example
python test_monorec.py
```

## 🗃️ Data

In all of our experiments we used the KITTI Odometry dataset for training. For additional evaluations, we used the KITTI, Oxford RobotCar, 
TUM Mono-VO and TUM RGB-D datasets. All datapaths can be specified in the respective configuration files. In our experiments, we put all datasets into a seperate folder ```../data```.

### KITTI Odometry

To setup KITTI Odometry, download the color images and calibration files from the 
[official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (around 145 GB). Instead of the given 
velodyne laser data files, we use the improved ground truth depth for evaluation, which can be downloaded from 
[here](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php). 

Unzip the color images and calibration files into ```../data```. The lidar depth maps can be extracted into the given 
folder structure by running ```data_loader/scripts/preprocess_kitti_extract_annotated_depth.py```.

For training and evaluation, we use the poses estimated by [Deep Virtual Stereo Odometry (DVSO)](https://vision.in.tum.de/research/vslam/dvso). They can be downloaded 
from [here](https://vision.in.tum.de/_media/research/monorec/poses_dvso.zip) and should be placed under ``../data/{kitti_path}/poses_dso``. This folder structure is ensured when 
unpacking the zip file in the ``{kitti_path}`` directory.

The auxiliary moving object masks can be downloaded from [here](https://vision.in.tum.de/_media/research/monorec/mvobj_mask.zip). They should be placed under 
``../data/{kitti_path}/sequences/{seq_num}/mvobj_mask``. This folder structure is ensured when 
unpacking the zip file in the ``{kitti_path}`` directory.

### Oxford RobotCar


To setup Oxford RobotCar, download the camera model files and the large sample from 
[the official website](https://robotcar-dataset.robots.ox.ac.uk/downloads/). Code, as well as, camera extrinsics need to be downloaded 
from the [official GitHub repository](https://github.com/ori-mrg/robotcar-dataset-sdk).
Please move the content of the ``python`` folder to ``data_loaders/oxford_robotcar/``.
``extrinsics/``, ``models/`` and ``sample/`` need to be moved to ``../data/oxford_robotcar/``. Note that for poses we 
use the official visual odometry poses, which are not provided in the large sample. They need to be downloaded manually from
[the raw dataset](http://mrgdatashare.robots.ox.ac.uk/download/?filename=datasets/2014-12-12-10-45-15/2014-12-12-10-45-15_vo.tar) 
and unpacked into the sample folder.

### TUM Mono-VO

Unfortunately, TUM Mono-VO images are provided only in the original, distorted form. Therefore, they need to be undistorted 
first before fed into MonoRec. To obtain poses for the sequences, we run the publicly available version 
of [Direct Sparse Odometry](https://github.com/JakobEngel/dso).

### TUM RGB-D

The official sequences can be downloaded from [the official website](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)
and need to be unpacked under ``../data/tumrgbd/{sequence_name}``. Note that our provided dataset implementation assumes 
intrinsics from ``fr3`` sequences. Note that the data loader for this dataset also relies on the code from the Oxford Robotcar dataset.

## 🏋️ Training & Evaluation

**Please stay tuned! Training code will be published soon!**

We provide checkpoints for each training stage:

| Training stage | Download |
| --- | --- |
| Depth Bootstrap  | [Link](https://vision.in.tum.de/_media/research/monorec/monorec_depth.pth.zip) |
| Mask Bootstrap  | [Link](https://vision.in.tum.de/_media/research/monorec/monorec_mask.pth.zip) |
| Mask Refinement  | [Link](https://vision.in.tum.de/_media/research/monorec/monorec_mask_ref.pth.zip) |
| Depth Refinement (**final model**)  | [Link](https://vision.in.tum.de/_media/research/monorec/monorec_depth_ref.pth.zip) |

Run ``download_model.sh`` to download the final model. It will automatically get moved to ``saved/checkpoints``. 

To reproduce the evaluation results on different datasets, run the following commands:

```shell
python evaluate.py --config configs/evaluate/eval_monorec.json        # KITTI Odometry
python evaluate.py --config configs/evaluate/eval_monorec_oxrc.json   # Oxford Robotcar
```

## ☁️ Pointclouds

To reproduce the pointclouds depicted in the paper and video, use the following commands:

```shell
python create_pointcloud.py --config configs/test/pointcloud_monorec.json       # KITTI Odometry
python create_pointcloud.py --config configs/test/pointcloud_monorec_oxrc.json  # Oxford Robotcar
python create_pointcloud.py --config configs/test/pointcloud_monorec_tmvo.json  # TUM Mono-VO
```