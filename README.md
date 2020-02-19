# SEAN: Image Synthesis with Semantic Region-Adaptive Normalization - Non-official PyTorch implementation
## Installation

Clone this repo.
```bash
git git@github.com:zhang-zx/SEAN-PyTorch.git
cd SEAN-PyTorch/
```

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

This code also requires the Synchronized-BatchNorm-PyTorch rep.
```
cd models/sean_networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

To reproduce the results reported in the paper, you would need an NVIDIA DGX1 machine with 8 V100 GPUs.

## Dataset Preparation

Download CelebaHQ_Mask dataset and place all unziped data in `./datasets/CelebAHQ_MASK`. After all data is placed correctly, go to `./datasets` and run `python CelebAHQ_Mask_parser.py`.

**Note:** you should set `cpu_count` to real num of your computer if you are running in a virtual machine.

## Training New Models

New models can be trained with the following commands.

```bash
# To train on CelebaHQ_Mask Dataset
#!/usr/bin/env bash
set -ex

label_dir=[mask_dir]
image_dir=[img_dir]
num_labels=19
batchSize=12
num_workers=10

experiment_name=[experiment name]

nohup python -u train.py --name ${experiment_name}  \
                    --dataset_mode custom \
                    --label_dir ${label_dir} \
                    --image_dir ${image_dir} \
                    --label_nc ${num_labels} \
                    --model pix2pix_sean \
                    --batchSize ${batchSize} \
                    --load_size 256 --crop_size 256 --label_nc ${num_labels} \
                    --gpu_ids 0,1,2,3,4,5 \
                    --no_instance \
                    --display_freq 500 \
                    --nThreads ${num_workers} --netG sean --tf_log > ${experiment_name}.log 2>&1&

tail -f ${experiment_name}.log
```

There are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `--gpu_ids`. If you want to use the second and third GPUs for example, use `--gpu_ids 1,2`.

To log training, use `--tf_log` for Tensorboard. The logs are stored at `[checkpoints_dir]/[name]/logs`.

## Testing

Testing is similar to testing pretrained models.

```bash
#!/usr/bin/env bash
set -ex

label_dir=[mask_dir]
image_dir=[img_dir]
num_labels=19
batchSize=8
num_workers=10
how_many=500

experiment_name=baseline

nohup python -u test.py --name ${experiment_name}  \
                    --dataset_mode custom \
                    --label_dir ${label_dir} \
                    --image_dir ${image_dir} \
                    --label_nc ${num_labels} \
                    --model pix2pix_sean \
                    --batchSize ${batchSize} \
                    --how_many ${how_many} \
                    --load_size 256 --crop_size 256 --label_nc ${num_labels} \
                    --gpu_ids 0,1,2,3 \
                    --no_instance \
                    --nThreads ${num_workers} --netG sean  > ${experiment_name}_test.log 2>&1&

tail -f ${experiment_name}_test.log
```

Use `--results_dir` to specify the output directory. `--how_many` will specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `trainers/pix2pix_sean_trainer.py`: harnesses and reports the progress of training.
- `models/pix2pix_sean_model.py`: creates the networks, and compute the losses
- `models/networks/`: defines the architecture of all models
- `options/`: creates option lists using `argparse` package. More individuals are dynamically added in other files as well. Please see the section below.
- `data/`: defines the class for loading images and label maps.

## Options

This code repo contains many options. Some options have different default values depending on other options. To address this, the `BaseOption` class dynamically loads and sets options depending on what model, network, and datasets are used. This is done by calling the static method `modify_commandline_options` of various classes. It takes in the`parser` of `argparse` package and modifies the list of options. For example, since COCO-stuff dataset contains a special label "unknown", when COCO-stuff dataset is used, it sets `--contain_dontcare_label` automatically at `data/coco_dataset.py`. You can take a look at `def gather_options()` of `options/base_options.py`, or `models/network/__init__.py` to get a sense of how this works.

## Acknowledgments
This code borrows heavily from SPADE.
