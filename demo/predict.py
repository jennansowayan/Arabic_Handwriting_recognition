'''
new_img_pred(clf, im_path='./3digit.jpeg')

The above method takes two parameters. clf is the trained model and im_path is the path of the image that is needed to be predicted.
For delivering a product, we should try a new data point, not in the dataset at all as follows:

-use an iPhone notes app
-draw an Arabic digit(3).
-import the image using OpenCV.
-convert image to black and white pixels.
-plot the image to visualize the digit.
-flip the image up down to meet the image orientation of the training dataset.
-resize the orginal image to 28x28 as in the dataset
-vectorizing the image
-then feed it to the model for prediction
-get the predicted class and verify with the plotted image.
'''

# import necessery liberyes for the project 
import pandas as pd
import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns

def new_img_pred(clf, im_path='./3digit.jpeg'):
    # import the image using OpenCV.
    img = cv2.imread(im_path)
    
    # convert image to black and white pixels.
    grayImage = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # plot the image to visualize the digit.
    plt.imshow(grayImage)
    plt.show()
    
    # flip the image up down to meet the image orientation of the training dataset.
    #grayImage = cv2.flip(grayImage,0)
    grayImage = cv2.rotate(grayImage, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.imshow(grayImage)
    plt.show()
    
    # resize the orginal image to 28x28 as in the dataset
    # dsize
    width  = 28
    height = 28
    dsize = (width, height)

    # resize image
    output = cv2.resize(grayImage, dsize, interpolation = cv2.INTER_AREA)
    plt.imshow(output)
    plt.show()
    
    # vectorizing the image
    vec_img = output.reshape(1, -1)
    
    # feed vector to the model for prediction
    # get the predicted class and verify with the plotted image.
    return "The predicted class is:"+str(clf.predict(vec_img)[0])
