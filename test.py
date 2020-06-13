import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from augmentators import randomHueSaturationValue, randomHorizontalFlip, randomShiftScaleRotate
from u_net import get_unet_128
import glob

import os

orig_width = 240
orig_height = 320

threshold = 0.5

epochs = 25
batch_size = 1
input_size, model = get_unet_128()

model_name = 'date-201912021649_epochs-25_batch-2_inputsize-128_net-normal.hdf5'

model_name_full = os.path.join('weights', model_name)

model.load_weights(model_name_full)

print(input_size)

test_filenames = glob.glob("input/test/*.png")
test_filenames = [filename.replace('\\', '/').replace('.png', '') for filename in test_filenames]
test_filenames = [filename.split('/')[-1] for filename in test_filenames]

print('Predicting on {} samples with batch_size = {}...'.format(len(test_filenames), batch_size))
for start in tqdm(range(0, len(test_filenames), batch_size)):
    x_batch = []
    end = min(start + batch_size, len(test_filenames))
    ids_test_batch = test_filenames[start:end]
    for id in ids_test_batch:
        img = cv2.imread('input/test/{}.png'.format(id))
        img = cv2.resize(img, (input_size, input_size))
        x_batch.append(img)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    for index, pred in enumerate(preds):
        prob = np.array(cv2.resize(pred, (orig_width, orig_height)) > threshold).astype(np.float32) * 255
        current_filename = ids_test_batch[index]
        cv2.imwrite('input/test/segmentation/{}.png'.format(id), prob)

print("Done!")
