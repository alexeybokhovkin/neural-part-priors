# Neural Part Priors
[![arXiv](https://img.shields.io/badge/arXiv-2203.09375-b31b1b.svg)](https://arxiv.org/abs/2203.09375)

Official code repository of "Neural Part Priors: Learning to Optimize Part-Based Object Completion in RGB-D Scans" @ CVPR 2023 (Highlight)

[Paper](https://arxiv.org/abs/2203.09375) | [Project website](https://alexeybokhovkin.github.io/neural-part-priors) |  [Video](https://www.youtube.com/watch?v=rWgFcFOy4LM)

![Teaser image](/static/teaser.svg)

## Method Overview

![Overview image](/static/overview.svg)

## Dependencies

* Tested on Ubuntu 20.04
* Python 3.8
* PyTorch 1.6.0
* CUDA 10.2

To install the full environment, use `conda` with our .yaml file:

```commandline
conda env create --file environment.yaml
conda activate torch
```

## Data processing

Our method combines using synthetic (ShapeNet/PartNet) data and real-world (ScanNet v2) data for training and inference. 

### Preparing synthetic data

1. To pretrain on synthetic data we first need to virtually scan the ShapeNet dataset (target categories) to produce complete 256^3 SDF grids. To perform this step we used [VirtualScan](https://github.com/angeladai/VirtualScan) library. After that we also virtually scanned the same ShapeNet shapes using only a portion of cameras to produce SDF grids of partially scanned objects. For every ShapeNet shape we generated 10 partial scans. This data is also needed for further processing and training.

2. Run ```src/dataproc/prepare_full_parts_uniform.py```. This script takes virtually scanned ShapeNet shapes, splits them into parts using PartNet data and prepares for DeepSDF training.

3. Run ```src/dataproc/prepare_full_shape_uniform.py```. This script takes virtually scanned ShapeNet shapes, splits them into parts using PartNet data and prepares for DeepSDF training.

4. Run ```src/dataproc/prepare_partial_voxels.py```. This script takes partial scans from the first step and produces voxel occupancy grids.

5. (Optional) Run ```src/dataproc/prepare_partial_parts_uniform.py```. This script takes partial scans from the first step, split them into PartNet parts and stores them in DeepSDF-suitable format. This step is optional, you can use this generated data to enhance the main model training.

### Preparing ScanNet data

1. First, we need to run [MLCVNET](https://github.com/NUAAXQ/MLCVNet) object detection method on ScanNet v2 data to detect furniture objects of target categories -- chair, table, bed, trashcan, cabinet, bookshelf. Follow the launch instructions in the official repository.

2. After that we need to extract SDF grids within bounding boxed detected with MLCVNet. To perform this step we developed the C++ code under ```src/dataproc/GenerateScans``` folder. The main logic is implemented in the ```src/dataproc/GenerateScans/VisualizerSN.cpp``` file and the main file is ```src/dataproc/GenerateScans/src/main.cpp```. To build this C++ project you need to include the following dependencies: [Boost 1.76.0](https://www.boost.org/users/history/version_1_76_0.html), [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page), [FreeImage](https://freeimage.sourceforge.io/), [json](https://github.com/nlohmann/json), [mLib](https://github.com/niessner/mLib) and [mLibExternal](https://www.dropbox.com/scl/fi/i327tt8hlsjly7x4u6r3z/mLibExternal.zip?rlkey=2qeu8x9n4laxq48nt2i1noqd0&e=1&dl=0).

3. Run ```src/dataproc/prepare_mlcvnet_points.py```. This script takes ScanNet scenes, bboxes detected with MLCVNet and stores points inside boxes splitted in parts. This data is necessary for the main model training (fine-tuning).

4. Run ```src/dataproc/prepare_mlcvnet_voxels.py```. This script takes as input MLCVNet points inside detected bounding boxes and voxelizes them. This data is necessary for the main model training (fine-tuning).

## Training
### Pretraining of the latent space and decoders
First, we need to pretrain our latent space for full shapes and their parts with corresponding decoders. Note that we train each pair of shape/part latent space and decoder for each shape category.

```commandline
cd deepsdf
bash launch_deepsdf_slurm.sh
```

This training script is based on the original [DeepSDF](https://github.com/facebookresearch/DeepSDF) training pipeline, but extends it with optional handling of shape parts and positional encoding. Thus, there are the following keys in the launch script:

```commandline
-e <a path to DeepSDF experiment folder containing specs.json and sv2_chair_train.json>
--batch_split <an integer number that specifies how many subbatches are contained in the batch for one training iteration>
--cat_name <a category name to train, see deepsdf/deep_sdf/data.py>
--learn_parts <specify this if you need to learn part embeddings>
--pe <specify this if you want to use positional encoding for 3D coordinates>
--continue <specify the name of a checkpoint if you want to continue the training>
```

We train both full shapes and shape parts until training curve is saturated. Our final loss values for all categories are ~2e-3 - 3e-3.

### Pretraining using synthetic data
Second, we use the pretrained embeddings from the previous state and learn mapping from synthetic incomplete voxel shapes to embeddings. Simultaneously we learn point cloud classification and parts prediction.

```commandline
cd scripts/executables
bash train_gnn_deepsdf_allshapes.sh
```

The script `train_gnn_deepsdf_allshapes.sh` uses config file `configs/config_train_partnet.yaml`. There are many parameters and paths to data, the description for the fields is the following:

```commandline
# paths & general configs
base: path to the folder with the NPPs project
datadir_trees: path to hierarchical PartNet trees (data is provided)
dataset: dataset_subfolder in datadir_trees
train_samples: .txt file with train samples
val_samples: .txt file with validation samples
checkpoint_dir: folder to store checkpoints and output data
version: name of subfolder in checkpoints_dir
cat_name: shape category to train

# auxiliary paths
full_shape_list_path: path to DeepSDF sv2_{cat_name}_train.json file with full shapes of a cat_name category
parts_list_path: path to DeepSDF sv2_{cat_name}_train.json file with parts of a cat_name category
mlcvnet_noise_path: path to a folder with ScanNet noise data of a cat_name category
data_mode: mode of training, partnet or mlcvnet
partnet_noise_dir: path to a folder with PartNet noise data of a cat_name category

# deep_sdf
sdf_data_source: path to SDF data which was used to train DeepSDF 
deepsdf_shape_path: path to DeepSDF full shape model checkpoint of a cat_name category
deepsdf_parts_path: path to DeepSDF part model checkpoint of a cat_name category
```

### Training using real data
Third, we finetine the model on real-world ScanNet data. The script `finetune_gnn_deepsdf_allshapes.sh` uses a config file `configs/config_train_mlcvnet.yaml`. There are many parameters and paths to data, which are similar to specified already in the pretraining step.

## Test-time optimization
### Straightforward TTO
Finally, the trained model is ready to be applied for test-time optimization to real-world scenes. The corresponding script `perform_deepsdf_tto_slurm.sh` uses a config file `configs/config_tto.yaml`.

### Scene-aware TTO
This part will be added later.

### Checkpoints and additional files

We provide the following checkpoints and files:

Latent codebook and decoder checkpoints (DeepSDF) -- [deepsdf_ckpt.zip](https://drive.google.com/file/d/1o9c34HxMAt7FhvUpb1ZdmMrJxgkTDp8r/view?usp=sharing),
Main model checkpoints -- [npps_ckpt.zip](https://drive.google.com/file/d/1dlltrvaUt1g1A12oEa227H1oo7d0PTzA/view?usp=sharing),
Meta data for DeepSDF training (chair objects, parts lists) -- [chair_lists.zip](https://drive.google.com/file/d/1sofS_y4K_mBOuO4Kuls9QQdBMIstkxSs/view?usp=sharing),
Voxelized training data (chair objects, ShapeNet/ScanNet) -- [chair_data.zip](https://drive.google.com/file/d/1NvzOvlCzgRkU6IT4HNgfFulRJrbxHU8d/view?usp=sharing).

## Citation
If you find our work useful, please cite using the following BibTex entry:
```
@inproceedings{bokhovkin2022neuralparts,
    title     = {Neural Part Priors: Learning to Optimize Part-Based Object Completion in RGB-D Scans},
    author    = {Bokhovkin, Alexey and Dai, Angela},
    journal   = {2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}
}
```



