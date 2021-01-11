# # from __future__ import print_function


import numpy as np
def mixup(batch_imgs,one_hot_batch_labels,beta=1,prob=1.):
    if np.random.uniform() < prob:
        batch_size = batch_imgs.shape[0]
        random_batch_indexes = np.random.choice(batch_size, batch_size,replace=False)
        mix_ratio_1 = np.random.beta(beta, beta)
        mix_ratio_2 = 1. - mix_ratio_1
        batch_imgs = mix_ratio_1 * batch_imgs+mix_ratio_2*batch_imgs[random_batch_indexes]
        # print(mix_ratio_1)
        one_hot_batch_labels = mix_ratio_1 * one_hot_batch_labels + mix_ratio_2 * one_hot_batch_labels[random_batch_indexes]
    return batch_imgs,one_hot_batch_labels
#
# import cv2
# IMG_WIDTH = 224
# IMG_HEIGHT = 224
# cv_img = cv2.imread('/home/wangem1/cam/cam_20210104/445.jpg')
# cv_img1 = cv2.imread('/home/wangem1/cam/cam_20210104/4.jpg')
# cv_img = cv2.resize(cv_img,(IMG_WIDTH,IMG_HEIGHT))
# cv_img1 = cv2.resize(cv_img1,(IMG_WIDTH,IMG_HEIGHT))
# batch_imgs = np.stack([cv_img,cv_img1])
# batch_labels = np.array([[0,1],[1,0]])
# i = 0
# while i<1:
#     batch_imgs,batch_labels=mixup(batch_imgs,batch_labels,1)
#     i+=1
#     print("testing:",i)
# cv2.imshow("d3",cv_img/255)
# cv2.imshow("d4",cv_img1/255)
# cv2.imshow("d1",batch_imgs[0]/255)
# cv2.imshow("d2",batch_imgs[1]/255)
# cv2.waitKey()
