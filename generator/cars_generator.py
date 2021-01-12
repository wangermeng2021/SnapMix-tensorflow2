# # from __future__ import print_function

import pandas as pd
import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from utils.snapmix import snapmix
from utils.cutmix import cutmix
from utils.mixup import mixup
from utils.baseline_augment import resize_and_random_crop
from utils.baseline_augment import resize_and_center_crop

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

from scipy.io import loadmat
def get_mat_frame(path, img_folder):
    results = {}
    tmp_mat = loadmat(path)
    anno = tmp_mat['annotations'][0]
    results['path'] = [os.path.join(img_folder, anno[i][-1][0]) for i in range(anno.shape[0])]
    results['label'] = [anno[i][-2][0, 0] for i in range(anno.shape[0])]
    return results

class CarsGenerator(tf.keras.utils.Sequence):
    def __init__(self, root_dir, mode='train', batch_size=32, augment='baseline'):
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

        self.augment = augment
        self.mode = mode
        self.batch_size = batch_size

        self.eppch_index = 0
        self.num_class = len(np.unique(self.label_list))

        self.data_index = np.arange(0, len(self.label_list))
        if self.mode == 'train':
            np.random.shuffle(self.data_index)

        self.resize_size = (512, 512)
        self.crop_size = (448, 448)

    def set_model(self,model):
        self.last_conv_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('last_conv').output)
        self.last_dense_layer = model.get_layer('predictions')
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
        return image
        # return image[:, :, ::-1]
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        self.eppch_index += 1
    def __len__(self):
        return len(self.img_path_list) // self.batch_size
    def __getitem__(self, batch_index):

        batch_img_paths = self.img_path_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = self.label_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = np.array(batch_labels)
        one_hot_batch_labels = np.zeros([self.batch_size, self.num_class])
        batch_imgs = []
        if self.mode == "valid":
            for i in range(self.batch_size):
                img = self.read_img(batch_img_paths[i])
                resized_img = resize_and_center_crop(img, self.resize_size, self.crop_size)
                batch_imgs.append(resized_img)
                one_hot_batch_labels[i, batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
        else:
            for i in range(self.batch_size):
                img = self.read_img(batch_img_paths[i]).astype(np.uint8)
                resized_img = resize_and_random_crop(img, self.resize_size, self.crop_size)
                batch_imgs.append(resized_img)
                one_hot_batch_labels[i, batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
            # # if np.random.rand(1) < 0.5:
            if self.augment == 'snapmix':
                batch_imgs, one_hot_batch_labels = snapmix(batch_imgs,one_hot_batch_labels,self.last_conv_model,self.last_dense_layer,self.crop_size,5)
            elif self.augment == 'cutmix':
                batch_imgs, one_hot_batch_labels = cutmix(batch_imgs,one_hot_batch_labels,3)
            elif self.augment == 'mixup':
                batch_imgs, one_hot_batch_labels = mixup(batch_imgs,one_hot_batch_labels,1)

        return batch_imgs, one_hot_batch_labels




