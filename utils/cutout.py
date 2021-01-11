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

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

from cam import get_cam
from snapmix import snapmix


run_kaggle = False

import albumentations as A
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np
import time
import os
import glob
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import cv2
import json

from PIL import Image

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

import albumentations as A
# try:
#     import albumentations as A
# except:
#     !pip install -U albumentations

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/kaggle/input/mycode')
import augment

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0],True)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

if run_kaggle:
    train_csv_path  = '../input/cassava-leaf-disease-classification/train.csv'
    train_image_dir = '../input/cassava-leaf-disease-classification/train_images/'
    test_image_dir  = '../input/cassava-leaf-disease-classification/test_images'
else:
    train_csv_path = '/home/wangem1/dataset/cassava-leaf-disease-classification/train.csv'
    train_image_dir = '/home/wangem1/dataset/cassava-leaf-disease-classification/train_images/'
    test_image_dir = '/home/wangem1/dataset/cassava-leaf-disease-classification/test_images'


class DataAugment():
    def __init__(self):
        pass

    def albumentations(self, img):
        transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),

            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),

            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),

            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),

            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),

            A.HueSaturationValue(p=0.3),
        ])

        return transform(image=img)['image']

    #         file = preprocess_input(file)

    def random_horizontal_flip(self, img):
        if np.random.uniform() > 0.5:
            return img[:, ::-1, :]

    def random_vertical_flip(self, img):
        if np.random.uniform() > 0.5:
            return img[::-1, :, :]

    def cut_mix(self, X1, Y1, X2, Y2):
        # return X2, Y2

        alpha = 1.0
        l = np.random.beta(alpha, alpha, (X1.shape[0], 3))
        # l = np.random.uniform(0.0, 1.0, (X1.shape[0], 3))
        ly_ratio = l[:, 0]
        lx_ratio = l[:, 1]
        larea_ratio = l[:, 2]

        larea_ratio = np.sqrt(larea_ratio)

        crop_h = (np.array(X1.shape[1]) * (larea_ratio)).astype(int)
        crop_w = (np.array(X1.shape[2]) * (larea_ratio)).astype(int)
        crop_y = (np.array(X1.shape[1]) * ly_ratio).astype(int)
        crop_x = (np.array(X1.shape[2]) * lx_ratio).astype(int)

        larea_ratio = (np.clip(crop_y + crop_h, 0, X1.shape[1]) - crop_y) * (
                np.clip(crop_x + crop_w, 0, X1.shape[2]) - crop_x) / (X1.shape[2] * X1.shape[1])
        Y2 = larea_ratio[..., np.newaxis] * Y1 + (1 - larea_ratio)[..., np.newaxis] * Y2

        for bi in range(X1.shape[0]):
            X2[bi, crop_y[bi]: crop_y[bi] + crop_h[bi], crop_x[bi]: crop_x[bi] + crop_w[bi], :] = \
                X1[bi, crop_y[bi]: crop_y[bi] + crop_h[bi], crop_x[bi]: crop_x[bi] + crop_w[bi], :]

        return X2, Y2
from scipy.io import loadmat
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_mat_frame(path, img_folder):
    results = {}
    tmp_mat = loadmat(path)
    anno = tmp_mat['annotations'][0]
    results['path'] = [os.path.join(img_folder, anno[i][-1][0]) for i in range(anno.shape[0])]
    results['label'] = [anno[i][-2][0, 0] for i in range(anno.shape[0])]
    return results

# root_dir = "/home/wangem1/cam/cam_20201231_2/dataset/car_dataset"
# mode="train"
# pd_train = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root_dir, 'devkit', 'cars_train_annos.mat'), 'cars_train'))
#
# pd_test = pd.DataFrame.from_dict(
#     get_mat_frame(os.path.join(root_dir, 'devkit', 'cars_test_annos_withlabels.mat'), 'cars_test'))
# data = pd.concat([pd_train, pd_test])
#
# data['train_flag'] = pd.Series(data.path.isin(pd_train['path']))
# data = data[data['train_flag'] == (mode == 'train')]
# data['label'] = data['label'] - 1
# imgs = data.reset_index(drop=True)
# if len(imgs) == 0:
#     raise (RuntimeError("no csv file"))
#
# s1 = root_dir+imgs['path'].values
# print(s1)
# exit()


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, root_dir, mode='train', kfold_index=0, batch_size=32,img_size=(224, 224),model=None):

        pd_train = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root_dir, 'devkit', 'cars_train_annos.mat'), 'cars_train'))
        pd_test = pd.DataFrame.from_dict(get_mat_frame(os.path.join(root_dir, 'devkit', 'cars_test_annos_withlabels.mat'), 'cars_test'))
        data = pd.concat([pd_train, pd_test])
        data['train_flag'] = pd.Series(data.path.isin(pd_train['path']))

        data = data[data['train_flag'] == (mode == 'train')]
        data['label'] = data['label'] - 1
        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise(RuntimeError("no csv file"))

        full_path_list = root_dir+"/"+imgs['path'].values
        self.img_path_list =full_path_list
        self.label_list = imgs['label'].values


        self.last_conv_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('last_conv').output)
        self.last_dense_layer = model.get_layer('predictions')

        self.mode = mode
        self.data_aug = DataAugment()
        self.auto_aug = augment.RandAugment()
        self.batch_size = batch_size
        self.img_size = np.array(img_size)

        self.eppch_index = 0
        self.num_class = len(np.unique(self.label_list))

        self.data_index = np.arange(0,len(self.label_list))
        if self.mode == 'train':
            np.random.shuffle(self.data_index)

    def resize(self, img, size):
        img_width_height = img.shape[0:2][::-1]
        max_ratio = np.max(img_width_height / size)
        img_width_height = img_width_height / max_ratio
        min_side = np.min(img_width_height).astype(np.int32)
        dst_size = [[min_side, size[1]], [size[0], min_side]][np.argmin(img_width_height)]
        resized_img = cv2.resize(img, tuple(dst_size))
        pad_height = (size[1] - resized_img.shape[0]) // 2
        pad_width = (size[0] - resized_img.shape[1]) // 2
        return np.pad(resized_img, [[pad_height, size[0] - pad_height - resized_img.shape[0]],
                                    [pad_width, size[1] - pad_width - resized_img.shape[1]], [0, 0]])
    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image[:, :, ::-1]
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        self.eppch_index+=1
    def __len__(self):
        # return 10
        return len(self.img_path_list) // self.batch_size

    def __getitem__(self, batch_index):

        batch_img_paths = self.img_path_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = self.label_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = np.array(batch_labels)
        one_hot_batch_labels = np.zeros([self.batch_size, self.num_class])
        batch_imgs = []
        if self.mode == "valid":
            for i in range(self.batch_size):
                # resized_img = cv2.imread(batch_img_paths[i])
                resized_img = self.read_img(batch_img_paths[i])
                resized_img = self.resize(resized_img,np.array([256,256]))
                transform = A.Compose([
                    A.CenterCrop(width=224, height=224),
                ])
                resized_img = transform(image=resized_img)['image']
                # resized_img = resized_img / 127.5-1.

                # x = resized_img[..., ::-1]
                resized_img = resized_img.astype(np.float32)
                mean = [103.939, 116.779, 123.68]
                resized_img[..., 0] -= mean[0]
                resized_img[..., 1] -= mean[1]
                resized_img[..., 2] -= mean[2]

                # resized_img = resized_img/255.
                batch_imgs.append(resized_img)
                one_hot_batch_labels[i, batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
        else:
            for i in range(self.batch_size):
                # resized_img = cv2.imread(batch_img_paths[i]).astype(np.uint8)
                resized_img = self.read_img(batch_img_paths[i]).astype(np.uint8)

                # resized_img = self.data_aug.albumentations(resized_img)
                # resized_img =  self.auto_aug.distort(tf.constant(resized_img)).numpy()

                resized_img = self.resize(resized_img,np.array([256,256]))
                transform = A.Compose([
                    A.RandomCrop(width=224, height=224),
                    A.HorizontalFlip(p=0.5),
                    # A.RandomBrightnessContrast(p=0.2),
                ])
                resized_img = transform(image=resized_img)['image']
                # resized_img = resized_img / 127.5-1.
                resized_img = resized_img.astype(np.float32)
                mean = [103.939, 116.779, 123.68]
                resized_img[..., 0] -= mean[0]
                resized_img[..., 1] -= mean[1]
                resized_img[..., 2] -= mean[2]
                # resized_img = resized_img/255.
                # np.asarray(resized_img).astype('float32')
                # print(resized_img)
                # cv2.imshow("d",resized_img)
                # cv2.waitKey()
                # plt.imshow(resized_img)
                # plt.show()

                batch_imgs.append(resized_img)
                one_hot_batch_labels[i, batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
            # # if np.random.rand(1) < 0.5:
            # if self.eppch_index>5:
            batch_imgs,one_hot_batch_labels = snapmix(batch_imgs,one_hot_batch_labels,self.last_conv_model,self.last_dense_layer,tuple(self.img_size),5)

            # test_img = cv2.imread("/home/wangem1/cam/cam_20201231_2/445.jpg")
            # test_img = cv2.resize(test_img, (224, 224))
            # test_img = np.expand_dims(test_img, axis=0)
            # conv_out = self.last_conv_model.predict(test_img)
            # car_cam = get_cam(conv_out, self.last_dense_layer,np.array([1]),tuple(self.img_size))
            # car_cam = car_cam * 255
            # car_cam = np.expand_dims(car_cam, -1)
            # car_cam = car_cam.astype(np.uint8)
            # cv2.imwrite("123/%f.jpg"%(time.time()),car_cam[0])



            # half_batch_size = self.batch_size//2
            # batch_imgs,one_hot_batch_labels = self.data_aug.cut_mix(batch_imgs[0:half_batch_size],one_hot_batch_labels[0:half_batch_size],
            #                                                         batch_imgs[half_batch_size:],one_hot_batch_labels[half_batch_size:])


        return batch_imgs, one_hot_batch_labels






def warmup_lr_scheduler(warmup_learning_rate=0.00001,warmup_epochs=0,init_learning_rate=0.0001,epochs=10):
    warmup_lr = warmup_learning_rate
    warmup_epochs = warmup_epochs
    init_learning_rate = init_learning_rate
    epochs = epochs

    def scheduler(epoch, lr):

        if epoch < warmup_epochs:
            current_epoch_lr = warmup_lr+epoch*(init_learning_rate-warmup_lr)/warmup_epochs
        else:
            current_epoch_lr = init_learning_rate*(1.0+tf.math.cos(np.pi/(epochs-warmup_epochs)*(epoch-warmup_epochs)))/2.0
        print(epoch,current_epoch_lr)
        return current_epoch_lr
    return scheduler



'''
Base model	resolution
EfficientNetB0	224
EfficientNetB1	240
EfficientNetB2	260
EfficientNetB3	300
EfficientNetB4	380
EfficientNetB5	456
EfficientNetB6	528
EfficientNetB7	600
'''
import tempfile
def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
    # Add L1L2 regularization to the whole model.
    # NOTE: This will save and reload the model. Do not call this function inplace but with
    # model = add_l1l2_regularizer(model, ...)

    if not reg_attributes:
        # reg_attributes = ['kernel_regularizer', 'bias_regularizer',
        #                   'beta_regularizer', 'gamma_regularizer']
        reg_attributes = ['kernel_regularizer']
    if isinstance(reg_attributes, str):
        reg_attributes = [reg_attributes]

    regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)
            if hasattr(layer, 'bias_regularizer'):
                setattr(layer, 'bias_regularizer', None)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # Reload the model
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)
    return model


def get_model(num_class=2):
    def concat_max_average_pool(x):
        return x
        x1 = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x2 = tf.keras.layers.AveragePooling2D((2, 2))(x)
        return tf.concat([x1, x2], axis=-1)
    if False:
        import tensorflow_hub as hub
        module = hub.KerasLayer("models/bit_m-r50x1_1", trainable=True)
        input_layer = tf.keras.Input((None,None,3))
        x = module(input_layer)
        # x = tf.keras.layers.Dense(1024,activation='relu')(features)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(num_class,activation='softmax')(x)
        # x = tf.keras.layers.Dense(num_class)(x)
        # x = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

        model = tf.keras.Model(inputs=input_layer,outputs=x)
        return model
    else:

        input_layer = tf.keras.Input(shape=(None, None, 3))
        # x = tf.keras.applications.EfficientNetB0(include_top=False, weights="efficientnetb0_notop.h5")(input_layer)
        x = tf.keras.applications.ResNet50(include_top=False, weights="/home/wangem1/cam/cam_20210104/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")(input_layer)
        # x = tf.keras.applications.EfficientNetB0(include_top=False, weights=None)(input_layer)
        # model = tf.keras.Model(inputs=input_layer,outputs=x)

        x = tf.keras.layers.Lambda(concat_max_average_pool,name="last_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(num_class, activation='softmax',name="predictions")(x)

        # x = tf.keras.layers.Dense(num_class)(x)
        # x = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

        model = tf.keras.Model(inputs=input_layer, outputs=x)
        # model = tf.keras.models.load_model('/home/wangem1/classification/classification/checkpoints/kfold:0.weights.03-0.4035-0.8628')

        return model


#
# #
# model = get_model(196)
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# BATCH_SIZE=4
# kfold=0
# train_generator = DataGenerator(root_dir='/home/wangem1/cam/cam_20210104/dataset/car_dataset',mode="valid",batch_size=BATCH_SIZE,img_size=(IMG_HEIGHT, IMG_WIDTH),kfold_index=kfold,model=model)
# for batch_imgs,one_hot_batch_labels in train_generator:
#
#     print("t1:", one_hot_batch_labels[0])
#     print("t2:", one_hot_batch_labels[1])
#     print("t3:", one_hot_batch_labels[2])
#     print("t4:", one_hot_batch_labels[3])
#     cv2.imshow("b1", batch_imgs[0])
#     cv2.imshow("b2", batch_imgs[1])
#     cv2.imshow("b3", batch_imgs[2])
#     cv2.imshow("b4", batch_imgs[3])
#     cv2.waitKey()
#
# exit()


for kfold in range(1):
    tf.keras.backend.clear_session()

    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 64
    epochs = 100
    init_lr = 0.0001
    # weight_decay = 5e-4
    weight_decay = 0.

    print("kfold:{} start running!".format(kfold))
    model = get_model(196)

    if weight_decay > 0.0:
        # for layer in model.layers:
        #     if hasattr(layer, 'kernel_regularizer'):
        #         layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
        model = add_l1l2_regularizer(model, l2=weight_decay)
        print("adding weight decay!")
    else:
        print('no weight decay!')
    callbacks = []
    callbacks.append(
        tf.keras.callbacks.LearningRateScheduler(warmup_lr_scheduler(init_learning_rate=init_lr, epochs=epochs)))
    # callbacks.append(tf.keras.callbacks.ModelCheckpoint(
    #     os.path.join('checkpoints', 'kfold:%d.weights.{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}'%kfold), save_best_only=True,
    #     save_weights_only=False))
    # callbacks.append(tf.keras.callbacks.EarlyStopping(patience=30, verbose=1))
    # callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(verbose=1))
    # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir='logs'))
    train_generator = DataGenerator(root_dir='/home/wangem1/cam/cam_20210104/dataset/car_dataset',mode="train",batch_size=BATCH_SIZE,img_size=(IMG_HEIGHT, IMG_WIDTH),kfold_index=kfold,model=model)
    valid_generator = DataGenerator(root_dir='/home/wangem1/cam/cam_20210104/dataset/car_dataset',mode="valid",batch_size=BATCH_SIZE,img_size=(IMG_HEIGHT, IMG_WIDTH),kfold_index=kfold,model=model)







    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0), optimizer=tf.keras.optimizers.SGD(learning_rate=init_lr,momentum=0.9),metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0), optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),metrics=['accuracy'])
    model.fit(train_generator, validation_data=valid_generator,epochs=epochs,callbacks=callbacks)

    # train_generator.img_size = np.array((260,260))
    # train_generator.batch_size = 12
    # model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
    #               metrics=['accuracy'])
    # model.fit(train_generator, validation_data=valid_generator, epochs=3, callbacks=callbacks, workers=4)
    #
    # train_generator.img_size = np.array((380, 380))
    # train_generator.batch_size = 8
    # model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
    #               metrics=['accuracy'])
    # model.fit(train_generator, validation_data=valid_generator, epochs=10, callbacks=callbacks, workers=4)

    del train_generator
    del valid_generator
    del model