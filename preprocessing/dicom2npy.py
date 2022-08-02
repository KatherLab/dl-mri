import cv2
import os
import SimpleITK as sitk
import numpy as np
from natsort import natsorted
import pandas as pd

'''
For unet training
'''


def unet_train_generator(mri_root, l_start=0, l_end=10):
    X_train = []
    Y_train = []
    id_list = []

    p_list = list(os.listdir(mri_root))
    if '.DS_Store' in p_list:
        p_list.remove('.DS_Store')

    remove_list = ['pre-A54', 'pre-B262', 'pre-B104', 'pre-B308', 'pre-C9']  # which mask is wrong
    for rmf in remove_list:
        p_list.remove(rmf)

    for i, id in enumerate(p_list[l_start:l_end]):
        p_path = os.path.join(mri_root, id)
        mask_file = os.path.join(p_path, 'N' + id + '.nii')
        mask = sitk.ReadImage(mask_file)
        mask_array = sitk.GetArrayFromImage(mask)

        mri_path = os.path.join(p_path, 'sort_t2')
        # print(id)
        mri_list = natsorted(os.listdir(mri_path))

        for h in range(mask_array.shape[0]):
            # if mask_array[h, ...].sum() > 100:
            if 1:
                mask_single_array = mask_array[h, :, :]
                mri_file = os.path.join(mri_path, mri_list[-h - 1])
                mri = sitk.ReadImage(mri_file)
                mri_array = sitk.GetArrayFromImage(mri)
                mri_single_array = mri_array[0, ...]

                mask_1 = cv2.resize(mask_single_array, (128, 128), interpolation=cv2.INTER_CUBIC)
                mri_1 = cv2.resize(mri_single_array, (128, 128), interpolation=cv2.INTER_CUBIC)

                X_train.append(np.expand_dims(mri_1, axis=-1))
                Y_train.append(np.expand_dims(mask_1, axis=-1))
                id_list.append(id)

    return X_train, Y_train, id_list


'''
Tumour segmentation for snet training
'''


def unet_predict_generator(mri_root, name_file_path, l_start=0, l_end=10):
    ori_img = {}
    resize_img = {}
    id_list = []

    name_df = pd.read_csv(name_file_path)
    p_list = ['pre-' + i for i in name_df.id]

    remove_list = ['pre-A54', 'pre-B262', 'pre-B104', 'pre-B308', 'pre-C9']  # which mask is wrong
    for rmf in remove_list:
        if rmf in p_list:
            p_list.remove(rmf)

    for i, id in enumerate(p_list[l_start:l_end]):
        p_path = os.path.join(mri_root, id)
        mri_path = os.path.join(p_path, 'sorted_t2')
        # print(id)
        # mri_list = sorted(os.listdir(mri_path), key=lambda s: int(s.split('-')[1].split('.')[0]))
        mri_list = natsorted(os.listdir(mri_path))
        mri_array = np.zeros([len(mri_list), 512, 512])  # for snet
        mri_resize_array = np.zeros([len(mri_list), 128, 128])  # for unet
        for j, mri_file in enumerate(mri_list):
            mri = sitk.ReadImage(os.path.join(mri_path, mri_file))
            mri_1 = sitk.GetArrayFromImage(mri)
            mri_single_array = mri_1[0, ...]
            mri_array[j, ...] = mri_single_array

            mri_resize = cv2.resize(mri_single_array, (128, 128), interpolation=cv2.INTER_CUBIC)
            mri_resize_array[j, ...] = mri_resize

        ori_img[id] = mri_array
        resize_img[id] = mri_resize_array
        id_list.append(id)

    return ori_img, resize_img, id_list
