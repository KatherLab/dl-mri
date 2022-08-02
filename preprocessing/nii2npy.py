import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from natsort import natsorted


def max_layer(array):
    h = array.shape[0]
    m_l = [np.sum(array[i, :, :]) for i in range(h)]
    l = m_l.index(max(m_l))
    return l


clini_df_path = '/input/clini.csv'
clini_df = pd.read_csv(clini_df_path)

for id in clini_df.id:
    ori_path = clini_df[clini_df.id == id]['ori_path'].values[0]
    dcm_list = os.listdir(ori_path)
    mask_path = clini_df[clini_df.id == id]['mask_path'].values[0]
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    sel_layer = max_layer(mask_array)

    mask_img = mask_array[sel_layer, :, :]

    mri_list = natsorted(dcm_list)
    mri_file = os.path.join(ori_path, mri_list[-sel_layer - 1])  # upside-down
    mri = sitk.ReadImage(mri_file)
    mri_array = sitk.GetArrayFromImage(mri)
    mri_img = mri_array[0, :, :]

    file_name = id + '.npy'

    np.save(os.path.join('output/mask', file_name), mask_img)
    np.save(os.path.join('output/ori', file_name), mri_img)
