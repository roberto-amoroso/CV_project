import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from augmentators import randomHueSaturationValue, randomHorizontalFlip, randomShiftScaleRotate
from u_net import get_unet_128, get_unet_128_modified
import glob
from datetime import datetime
import time

epochs = 1
batch_size = 4  # 2 or 4
input_size, model = get_unet_128()
# input_size, model = get_unet_128(input_shape=(256, 256, 3))
# model.load_weights(filepath='weights/best_weights.hdf5') # For resuming train

weigth_name = datetime.now().strftime('weights/%Y_%m_%d_%H_%M_') + (
            'epochs_%d_batch_%d_insize_%d_weights.hdf5' % (epochs, batch_size, input_size))

train_img_path_template = 'MyDataset/train/{}.jpg'
train_img_mask_path_template = 'MyDataset/train/segmentation/{}.png'

train_filenames = glob.glob("MyDataset/train/*.jpg")
train_filenames = [filename.replace('\\', '/').replace('.jpg', '') for filename in train_filenames]
train_filenames = [filename.split('/')[-1] for filename in train_filenames]

train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

print('Training on {} samples'.format(len(train_split)))
print('Validating on {} samples'.format(len(valid_split)))


def train_generator():
    while True:
        train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

        for start in range(0, len(train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(train_split))
            ids_train_batch = train_split[start:end]
            for id in ids_train_batch:
                # grayscale
                img = cv2.imread(train_img_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                img = np.dstack((img, img, img))

                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread(train_img_mask_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-55, 55),
                                               sat_shift_limit=(-10, 10),
                                               val_shift_limit=(-20, 20))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.30, 0.30),
                                                   scale_limit=(-0.35, 0.35),
                                                   rotate_limit=(-15, 15))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        train_split, valid_split = train_test_split(train_filenames, test_size=0.10, random_state=42)

        for start in range(0, len(valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(valid_split))
            ids_valid_batch = valid_split[start:end]
            for id in ids_valid_batch:
                # grayscale
                img = cv2.imread(train_img_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                img = np.dstack((img, img, img))

                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread(train_img_mask_path_template.format(id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


callbacks = [
    #        EarlyStopping(monitor='val_dice_loss',
    #                           patience=8,
    #                           verbose=1,
    #                           min_delta=1e-4,
    #                           mode='max'),
    ReduceLROnPlateau(monitor='val_dice_loss',
                      factor=0.5,
                      patience=4,
                      verbose=1,
                      epsilon=1e-5,
                      mode='max'),
    ModelCheckpoint(monitor='val_dice_loss',
                    filepath=weigth_name,
                    save_best_only=True,
                    save_weights_only=True,
                    mode='max'),
    TensorBoard(log_dir='logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(valid_split)) / float(batch_size)))