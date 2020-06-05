# EE290T Final Project

About
=====
This repository contains scripts that perform end-to-end segmentation on a point cloud that was computed with from source images with Pix4D. Segmentation is performed on the source images, and a transformation matrix provided by Pix4D us used to project the segmented pixels to points.

These scripts were written as part of a UC Berkeley EE290T final project by James Dunn and Sam Fernandes.

Getting Started
===============
Hardware
--------
Requires a host with CUDA, CUDNN, python3, PyTorch installed. Generally requires decent system and GPU memory but can be worked around with input image size (see below).

Photogrammetry Outputs
----------------------
This project assumes that the user has ran Pix4D and generated the following outputs:
* `.las` point cloud file
* `undistorted images` directory
* `offset.txt` and `pmatrix.txt`, matrices that capture the internal and external camera parameters.

For information on Pix4D outputs, see [this page](https://support.pix4d.com/hc/en-us/articles/202977149-What-does-the-Output-Params-Folder-contain)

Processing Steps
----------------

1. Check out this repository:
```
git clone git@github.com:jamesdunn/290t_project.git
```
2. Update submodules:
```
git submodule update --init --recursive
```
3. Place `project.las`, `pmatrix.txt`, and `offset.txt` in the root directory of this project.
4. Place all undistorted images in `unifiedparsing/input_images`
5. Run `PROJECT_DIR=$PWD` and `UPP_DIR=$PROJECT_DIR/unifiedparsing` in the shell (bash).
6. Run `source run.sh 1000`

The argument to `run.sh` is the width that input images should be downscaled to for Unified Parsing. Aspect ratio of images is maintained. Images used for photogrammetry are necessarily large, and such large images take enormous amounts of GPU memory when processed with Unified Parsing. This option downscales the input to Unified Parsing and accounts for the downscaling factor when projecting segmented image pixels to point cloud points.

Running PointNet++ on Data
==========
1. Ensure that your environment consists of the specific versions CUDA 9.0, CUDNN 7.1.4. tensorflow 1.12.0, and tensorflow-gpu 1.12.0.
2. Follow steps 1-5 in the Open3D-PointNet2-Semantic3D README to train PointNet2 on the Semantic3D dataset.
3. Place the point cloud to evaluate in `Open3D-PointNet2-Semantic3D/dataset` in [yourdata].txt, format, with columns
```
X Y Z intensity R G B
```
4. Place ground truth label for your dataset as a plaintext file named [yourdata].label, with rows consisting of per-point numeric class labels.
5. Add a dictionary containing your evaluation data file prefixes to `Open3D-PointNet2-Semantic3D/dataset/semantic_dataset.py`, following the format of the existing evaluation file prefix dictionary on line 28.
6. In the Open3D-PointNet2-Semantic3D submodule, run:
```
python predict.py --ckpt log/semantic/best_model_epoch_480.ckpt --set=[yourdata] --num_samples=500
```
7. Interpolate the sparse label results:
```
python interpolate.py --set=[yourdata]
```
The interpolated labels will be stored in `Open3D-PointNet2-Semantic3D/result/dense`, as a plaintext file with rows of per-point numeric Semantic3D classification labels. The framework will also generate accuracy values and confusion matrices for your data based on supplied ground truth.

Questions?
==========
Contact [my github username] at berkeley.edu.



