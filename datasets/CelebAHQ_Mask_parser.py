import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import numpy as np
from tqdm import tqdm

cpu_count = 40

root_dir = './datasets/CelebAHQ_MASK/CelebAMask-HQ-mask-anno'
out_dir = './datasets/CelebAHQ_MASK/Mask'
os.makedirs(out_dir, exist_ok=True)

label_dict = {
    'skin': 1, 'nose': 2, 'eye_g': 3, 'l_eye': 4, 'r_eye': 5, 'l_brow': 6, 'r_brow': 7, 'l_ear': 8,
    'r_ear': 9, 'mouth': 10, 'u_lip': 11, 'l_lip': 12, 'hair': 13, 'hat': 14, 'ear_r': 15, 'neck_l': 16,
    'neck': 17, 'cloth': 18
}

mask_dict = dict()


def process(new_path, mask_list):
    new_mask = np.zeros((512, 512))
    for mask_path, label in mask_list:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[0] != 512 or mask.shape[1] != 512:
            mask = cv2.resize(mask, (512, 512), cv2.INTER_CUBIC)
        mask = mask / 255.
        new_mask[((mask > 0.5).astype(np.uint8) + (new_mask < label).astype(np.uint8)) == 2] = label
    cv2.imwrite(new_path, new_mask.astype(np.uint8))


for root, _, files in os.walk(root_dir):
    for file in files:
        if file.find('.png') == -1:
            continue
        img_path = os.path.join(root, file)
        img_name = file[:file.find('_')] + '.png'
        img_key = file[file.find('_') + 1: -4]
        assert label_dict.__contains__(img_key), img_key
        if mask_dict.__contains__(img_name):
            mask_dict[img_name].append([img_path, label_dict[img_key]])
        else:
            mask_dict[img_name] = list()
            mask_dict[img_name].append([img_path, label_dict[img_key]])

executor = ProcessPoolExecutor(max_workers=cpu_count - 4)
futures = []
for key, val in tqdm(mask_dict.items()):
    new_img_path = os.path.join(out_dir, key)
    if os.path.exists(new_img_path):
        continue
    futures.append(executor.submit(
        partial(process, new_img_path, val)))
