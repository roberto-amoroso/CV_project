selector = int(input('Video source:\n0 - webcam (default)\n1 - video\n2 - video_2\n:') or '0')
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from weight import Weight
from augmentators import randomHueSaturationValue, randomHorizontalFlip, randomShiftScaleRotate
from u_net import *
import glob
from os import listdir
from os.path import isfile, join

orig_width = 340
orig_height = 260

threshold = 0.5

# date-201911271535_epochs-25_batch-2_inputsize-128_net-small
# date-201911271552_epochs-30_batch-2_inputsize-128_net-small

date-201911271449_epochs-20_batch-2_inputsize-192_net-small

weights_dir = 'weights/'

onlyfiles = [f for f in listdir(weights_dir) if isfile(join(weights_dir, f))]
for counter, value in enumerate(onlyfiles):
    try:
        w = Weight(value)
        print('%d - %s' % (counter, value))
    except:
        print('%d - NOT VALID %s' % (counter, value))

index_weights = int(input('Which weights open? ') or '0')

w = Weight(onlyfiles[index_weights])
weights = w.name
params = w.params

input_size = params['inputsize']
input_shape = (input_size, input_size, 3)

if params['net'] == 'normal': # normal
    input_size, model = get_unet_128(input_shape=input_shape)
elif params['net'] == 'big': # big
    input_size, model = get_unet_128_modified(input_shape=input_shape)
elif params['net'] == 'small': # small
    input_size, model = get_unet_128_small(input_shape=input_shape)
else:
    raise ValueError('Unkonw net:' + params['net'])

# model.load_weights('weights/2019_11_27_14_49_epochs_20_batch_2_inputsize_192_weights.hdf5')
model.load_weights(weights_dir + weights)

cv2.namedWindow('Camera Output')

# Get pointer to video frames from primary device
if selector == 0:
    videoFrame = cv2.VideoCapture(0)
elif selector == 1:
    videoFrame = cv2.VideoCapture('./data/test_video.mp4')
elif selector == 2:
    videoFrame = cv2.VideoCapture('./data/test_video_2.mp4')
else:
    raise ValueError("You have to choose a number from the list")

# Process the video frames
keyPressed = -1  # -1 indicates no key pressed

id = 0
count = 0

speed = 4

while keyPressed < 0:

    ###
    # For be smooth, jump some frames
    if selector != 0: # is not webcam
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

    # original_img_grey= cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    # sourceImage = np.dstack((original_img_grey, original_img_grey, original_img_grey))

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

    # Apply the mask on the image
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
