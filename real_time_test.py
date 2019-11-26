import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from augmentators import randomHueSaturationValue, randomHorizontalFlip, randomShiftScaleRotate
from u_net import get_unet_128, get_unet_128_modified
import glob

orig_width = 340
orig_height = 260

threshold = 0.5

epochs = 10
batch_size = 1
input_size, model = get_unet_128()
model.load_weights('weights/2019_11_26_20_03_epochs_10_batch_1_weights.hdf5')

cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
videoFrame = cv2.VideoCapture(0)
# videoFrame = cv2.VideoCapture('./data/test_video.mp4')
# videoFrame = cv2.VideoCapture('./data/test_video_2.mp4')

# Process the video frames
keyPressed = -1  # -1 indicates no key pressed

id = 0
count = 0

speed = 4

while keyPressed < 0:

    ###
    # For be smooth, jump some frames
    if count < speed:
        count += 1
        _ = videoFrame.read()
        continue
    else:
        count = 0

    # Grab video frame, decode it and return next video frame
    readSucsess, sourceImage = videoFrame.read()

    orig_height, orig_width, channels = sourceImage.shape
    # print(orig_height, orig_width, channels)

    original_img = cv2.resize(sourceImage, (orig_width, orig_height))

    original_img_grey= cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    sourceImage = np.dstack((original_img_grey, original_img_grey, original_img_grey))
    

    sourceImage = cv2.resize(sourceImage, (input_size, input_size))

    sourceImage = np.array([sourceImage], np.float32) / 255

    pred = model.predict_on_batch(sourceImage)

    pred = np.squeeze(pred, axis=3)

    prob = np.array(cv2.resize(pred[0], (orig_width, orig_height)) > threshold).astype(np.float32) * 255

    # for r in range(orig_height):
    #     for c in range(orig_width):
    #         if prob[r][c] > 250:
    #             original_img[r][c] = [219, 235, 52]
    #             # for i in range(3):
    #             #     original_img[r][c][i] = prob[r][c]

    ###
    # List comprehension
    # original_img = np.array(
    #     [[219, 235, 52] if prob[r][c] > 250 else original_img[r][c][i] for r in range(orig_height) for c in
    #      range(orig_width) for i in range(3)]).reshape((orig_height, orig_width, 3))

    ###

    # flip the image horizontally
    original_img = cv2.flip(original_img, 1)
    prob = cv2.flip(prob, 1)

    

    # Numpy
    red = np.where(prob, 219, original_img[:, :, 0])
    green = np.where(prob, 235, original_img[:, :, 1])
    blue = np.where(prob, 52, original_img[:, :, 2])

    rgb = np.dstack((red, green, blue))

    # gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # out = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.int8)

    # gray = np.dstack((out, out, out))
    # Display the source image
    cv2.imshow('Camera Output', rgb)

    # Check for user input to close program
    keyPressed = cv2.waitKey(1)  # wait 1 milisecond in each iteration of while loop

# Close window and camera after exiting the while loop
cv2.destroyWindow('Camera Output')
videoFrame.release()
