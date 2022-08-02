import os
import cv2
import numpy as np


def gen_dataset(data_dir, id_list, clini_df, img_size=224, outline=20):

    mask_dir = data_dir + '/mask'
    ori_dir = data_dir + '/ori'

    img_list = []
    os_list = []
    os_event_list = []
    dfs_list = []
    dfs_event_list = []

    for id in id_list:

        X = np.load(os.path.join(ori_dir, id))
        m = np.load(os.path.join(mask_dir, id))

        X = np.uint8(cv2.normalize(X, None, 0, 255, cv2.NORM_MINMAX))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        X = clahe.apply(X)
        X = (X - np.min(X)) / (np.max(X) - np.min(X))

        x, y = np.where(m > 0)

        w0, h0 = m.shape
        x_min = max(0, int(np.min(x) - outline))
        x_max = min(w0, int(np.max(x) + outline))
        y_min = max(0, int(np.min(y) - outline))
        y_max = min(h0, int(np.max(y) + outline))

        m = m[x_min:x_max, y_min:y_max]
        X = X[x_min:x_max, y_min:y_max]

        X_m_1 = X.copy()

        if X_m_1.shape[0] != img_size or X_m_1.shape[1] != img_size:
            X_m_1 = cv2.resize(X_m_1, (img_size, img_size), interpolation=cv2.INTER_CUBIC)


        X_m_1 = (X_m_1 - np.min(X_m_1)) / (np.max(X_m_1) - np.min(X_m_1))

        X_m_1 = np.expand_dims(X_m_1, axis=-1)
        X_m_1 = np.concatenate([X_m_1, X_m_1, X_m_1], axis=-1)


        img_list.append(X_m_1)
        os_list.append(clini_df[clini_df.id == id]['os'].values[0])
        os_event_list.append(clini_df[clini_df.id == id]['os_e'].values[0])
        dfs_list.append(clini_df[clini_df.id == id]['dfs'].values[0])
        dfs_event_list.append(clini_df[clini_df.id == id]['dfs_e'].values[0])

    return img_list, os_list, os_event_list, dfs_list, dfs_event_list

