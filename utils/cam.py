# # from __future__ import print_function
# from tensorflow import keras
# from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
# from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
import cv2
# from SpatialPyramidPooling import SpatialPyramidPooling
# from tensorflow.python.keras.datasets.cifar import load_batch
# from warmup import WarmupScheduler

# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0],True)


import matplotlib.pyplot as plt
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
#
#
# IMG_WIDTH = 224
# IMG_HEIGHT = 224
#
# # base_model = tf.keras.applications.EfficientNetB0(include_top=True, weights="/home/wangem1/classification/classification/efficientnetb0.h5",input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# base_model = tf.keras.applications.EfficientNetB0(include_top=True, weights="imagenet",input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# conv_model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.get_layer('top_activation').output)
#
# import cv2
# cv_img = cv2.imread("/home/wangem1/car.jpg")
# cv_img = cv2.resize(cv_img, (224, 224))
# cv2.imshow("ori",cv_img)
# cv_img = np.expand_dims(cv_img,axis=0)
#
# conv_out_ori = conv_model.predict(cv_img)
#

def get_cam(conv_out_ori,dense_layer,labels,img_size):
    filters, biases = dense_layer.get_weights()
    conv_out = np.reshape(conv_out_ori, (conv_out_ori.shape[0], -1, conv_out_ori.shape[-1]))
    filters = np.squeeze(filters)
    out1 = np.matmul(conv_out, filters)
    out2 = out1 + biases
    cam_out = np.reshape(out2, [np.shape(out2)[0], np.shape(conv_out_ori)[1], np.shape(conv_out_ori)[2], np.shape(out2)[-1]]).astype(np.float32)
    car_cam_list = []
    for i in range(labels.shape[0]):
        resized_cam_out = cam_out[i, :, :, labels[i]]
        resized_cam_out = cv2.resize(resized_cam_out,img_size,interpolation=cv2.INTER_LINEAR)
        car_cam_list.append(resized_cam_out)
    car_cam = np.stack(car_cam_list)

    # s1 = np.reshape(np.min(car_cam,axis=(1,2)),[np.min(car_cam,axis=(1,2)).shape[0],1,1])
    # s2 = np.max(car_cam, axis=(1, 2)) - np.min(car_cam, axis=(1, 2))
    # s2 = np.reshape(s2, [s2.shape[0], 1, 1])
    # car_cam = (car_cam - s1) /s2

    s1 = np.reshape(np.min(car_cam,axis=(1,2)),[car_cam.shape[0],1,1])
    s1 = car_cam - s1
    s2 = np.sum(s1, axis=(1, 2))
    s2 = np.reshape(s2, [s2.shape[0],1,1])
    car_cam = s1/s2


    return car_cam

# labels = np.array([751])
# dense_layer = base_model.get_layer('predictions')
#
# car_cam = get_cam(conv_out_ori,dense_layer,labels)
#
# car_cam = car_cam * 255
# car_cam = np.expand_dims(car_cam,-1)
# car_cam = car_cam.astype(np.uint8)
# cv2.imshow("B0",car_cam[0])
#
# cv2.waitKey()

#
# IMG_WIDTH = 224
# IMG_HEIGHT = 224
#
# cv_img = cv2.imread('/home/wangem1/cam/cam_20201231_2/445.jpg')
# cv_img1 = cv2.imread('/home/wangem1/cam/cam_20201231_2/4.jpg')
# cv_img = cv2.resize(cv_img,(IMG_WIDTH,IMG_HEIGHT))
# cv_img1 = cv2.resize(cv_img1,(IMG_WIDTH,IMG_HEIGHT))
# batch_imgs = np.stack([cv_img,cv_img1])
# batch_labels = np.array([1,0])
#
# # base_model = tf.keras.applications.EfficientNetB0(include_top=True, weights="/home/wangem1/classification/classification/efficientnetb0.h5",input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# base_model = tf.keras.applications.EfficientNetB0(include_top=True, weights="imagenet",input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
# last_conv_model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.get_layer('top_activation').output)
# last_dense_layer = base_model.get_layer('predictions')
# conv_out_ori = last_conv_model.predict(cv_img)
# car_cam = get_cam(conv_out_ori,last_dense_layer,batch_labels,(IMG_WIDTH,IMG_HEIGHT))
#
# car_cam = car_cam * 255
# car_cam = car_cam.astype(np.uint8)
# cv2.imshow("B0",car_cam[0])
# cv2.waitKey()
#
