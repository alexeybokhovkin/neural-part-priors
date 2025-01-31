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
This section will be updated later! 

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



