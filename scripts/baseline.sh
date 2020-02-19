#!/usr/bin/env bash
set -ex

label_dir=''
image_dir=''
num_labels=19
batchSize=12
num_workers=10

experiment_name=baseline

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