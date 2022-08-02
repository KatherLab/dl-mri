import numpy as np
import pandas as pd
from dataloader import gen_dataset
from vit import vit_b16

image_size = 224
model = vit_b16(
    image_size=image_size,
    activation=None,
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=1
)

model.load_weights('output/save_model/best_model.h5')

clini_df = pd.read_csv('input/valid.csv')
name_list = clini_df.id.tolist()

file_dir = 'valid/'
valid_img_list, valid_os_list, valid_os_event_list, \
valid_dfs_list, valid_dfs_event_list = gen_dataset(file_dir, name_list, clini_df)
valid_img = np.array(valid_img_list)

valid_score = -model.predict(valid_img)[:, 0]

pd.DataFrame({'id': name_list, 'score': valid_score}).to_csv('output/valid_score.csv')
