#!/usr/bin/env bash
set -ex

label_dir=''
image_dir=''
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
