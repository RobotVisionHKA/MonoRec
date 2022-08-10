# DenseReconstruction  
This is a clone of the [MonoRec](https://github.com/Brummi/MonoRec) repository with changes to run inference and train on Euroc type datasets, specifically the [TUM-VI](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) dataset.  

The primary additions are:  
1. Custom dataloader for tum-vi/euroc-format datasets  
2. Alternate script for viewing pointclouds using the [Open3D](http://www.open3d.org/docs/latest/index.html) library