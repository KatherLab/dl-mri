import numpy as np
import pandas as pd
import tensorflow as tf

from Early_Stopping import EarlyStopping
from dataloader import gen_dataset
from loss import cox_loss, concordance_index
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


clini_df = pd.read_csv('clini.csv')
name_list = clini_df.id.tolist()

X_train, X_test = train_test_split(name_list, test_size=0.3, random_state=42)


train_df = pd.DataFrame({'id': X_train, 'group': ["train" for i in X_train]})
test_df = pd.DataFrame({'id': X_test, 'group': ["test" for i in X_test]})
group_df = pd.concat([train_df, test_df])
group_df.to_csv('./group_df.csv')

file_dir = 'output/'
train_img_list, train_os_list, train_os_event_list, \
train_dfs_list, train_dfs_event_list = gen_dataset(file_dir, X_train, clini_df)
test_img_list, test_os_list, test_os_event_list, \
test_dfs_list, test_dfs_event_list = gen_dataset(file_dir, X_test, clini_df)

train_img = np.array(train_img_list)
test_img = np.array(test_img_list)
train_y = np.stack([train_os_list, train_os_event_list, train_dfs_list, train_dfs_event_list], axis=1).astype('float64')
test_y = np.stack([test_os_list, test_os_event_list, test_dfs_list, test_dfs_event_list], axis=1).astype('float64')

from vit import vit_b16
image_size = 224
model = vit_b16(
    image_size=image_size,
    activation=None,
    pretrained=True,
    include_top=True,
    pretrained_top=False,
    classes=1,
    dropout=0.1
)


datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

batch_size = 16
lr = 0.0001
epochs = 50
loss_ratio = [1, 1]
model_save_path = 'output/save_model'
early_stopping = EarlyStopping(model_path=model_save_path, patience=10, verbose=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    batches = 0
    for x_batch_train, y_batch_train in datagen.flow(train_img, train_y, batch_size=batch_size):
        batches += 1
        if batches >= train_img.shape[0] // batch_size:
            break
        with tf.GradientTape() as tape:

            y_pred = model(x_batch_train, training=True)

            os_loss = cox_loss(y_batch_train[:, 0:2], y_pred)
            dfs_loss = cox_loss(y_batch_train[:, 2:4], y_pred)
            loss_value = loss_ratio[0] * os_loss + loss_ratio[1] * dfs_loss
        grads = tape.gradient(loss_value, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_score = model.predict(train_img)
    test_score = model.predict(test_img)

    os_ci_train = concordance_index(train_y[:, 0:2], -train_score)
    os_ci_test = concordance_index(test_y[:, 0:2], -test_score)

    dfs_ci_train = concordance_index(train_y[:, 2:4], -train_score)
    dfs_ci_test = concordance_index(test_y[:, 2:4], -test_score)

    total_ci = os_ci_test + dfs_ci_test
    print(f'Test OS cindex:{os_ci_test}')
    print(f'Test DFS cindex:{dfs_ci_test}')

    early_stopping(-total_ci, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break
