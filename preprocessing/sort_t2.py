import os
import shutil
import pydicom
import pandas as pd

clini_df_path = ''

clini_df = pd.read_excel(clini_df_path)
path_list = clini_df.path.tolist()

for mri_path in path_list:
    id = clini_df[clini_df.path == mri_path].id.values[0]
    os.mkdir(os.path.join('/data/home/jiangxiaofeng/remote/github', 'data', id, 'sorted_t2'))
    sorted_fold = os.path.join('/data/home/jiangxiaofeng/remote/github', 'data', id, 'sorted_t2')

    dcm_list = os.listdir(mri_path)
    for dcm in dcm_list:
        file_path = os.path.join(mri_path, dcm)
        header = pydicom.dcmread(file_path)

        assert hasattr(header, 'InstanceNumber'), id + dcm + ' No InstanceNumber in dicom header'

        i_num = header.InstanceNumber

        new_path = os.path.join(sorted_fold, str(i_num) + '.dcm')
        shutil.copyfile(file_path, new_path)