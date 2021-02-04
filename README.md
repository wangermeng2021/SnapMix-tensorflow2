
# SnapMix-tensorflow2
A Tensorflow2.x implementation of SnapMix as described in [SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data](https://arxiv.org/abs/2012.04846)

## Features
- [x] mixup
- [x] cutmix
- [x] snapmix
- [x] resnet50,resnet101
- [x] efficientB0~B7
- [x] warmup
- [x] cosinedecay lr scheduler
- [x] step lr scheduler
- [x] concat-max-and-average-pool
- [x] custom dataset training

## Installation
###  1. Clone project
  ``` 
  git clone https://github.com/wangermeng2021/SnapMix-tensorflow2.git
  cd SnapMix-tensorflow2
  ```

###   2. Install environment
* install tesnorflow ( skip this step if it's already installed)
*     pip install -r requirements.txt

###   3. Download dataset
* Download cub dataset
  ```
  wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz -P dataset/
  tar -xzf dataset/CUB_200_2011.tgz -C dataset/
  mv dataset/CUB_200_2011 dataset/cub
  ```
* Download cars dataset
  ```
  wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz -P dataset/cars/
  wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz -P dataset/cars/
  wget https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P dataset/cars/
  tar -xzf  dataset/cars/cars_train.tgz -C dataset/cars
  tar -xzf  dataset/cars/cars_test.tgz -C dataset/cars
  tar -xzf  dataset/cars/car_devkit.tgz -C dataset/cars
  wget http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat -P dataset/cars/devkit/
  ```

## Training
* For training on cub dataset,use:
  ```
  python train.py --dataset cub --dataset-dir dataset/cub --model EfficientNetB0 --augment snapmix
  ```
* For training on Cars dataset,use:
  ```
  python train.py --dataset cars --dataset-dir dataset/cars --model EfficientNetB0 --augment snapmix
  ```
* For training on your custom dataset,use:
  ```
  python train.py --dataset custom --dataset-dir your_dataset_root_directory  --model EfficientNetB0  --augment snapmix
  ```
  you can try it on a toy dataset(No need to download dataset,it's already included in project:dataset/cat_dog):
  ```
  python train.py --dataset custom --dataset-dir dataset/cat_dog --model EfficientNetB0  --augment snapmix
  ```
  your_dataset_root_directory:  
  train  
 &nbsp; &nbsp; &nbsp; &nbsp; class1_name  
 &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; xxx.jpg  
 &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; xxx.jpg  
 &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; ...  
 &nbsp; &nbsp; &nbsp; &nbsp;class2_name  
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;xxx.jpg  
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;xxx.jpg  
 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;...  
 &nbsp; &nbsp;&nbsp; &nbsp; ...  
  valid  
  &nbsp; &nbsp;  &nbsp; &nbsp;class1_name  
  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; xxx.jpg  
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;xxx.jpg  
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;...  
  &nbsp; &nbsp; &nbsp; &nbsp;class2_name  
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;xxx.jpg  
  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; xxx.jpg  
  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;...  
  &nbsp; &nbsp; &nbsp; &nbsp;...  
## Evaluation results(RTX2080,epochs=300,batch_size=16):

| model                  |  cat_dog  | cars | cub  |
|------------------------|-----------|------|------|
| resnet50+cutmix        |  0.958    |      |      |
| resnet50+snapmix       |  0.979    |      |      |
| EfficientNetB0+mixup   |  0.968    |      |      |
| EfficientNetB0+cutmix  |  0.979    |      |      |
| EfficientNetB0+snapmix |  0.979    |      |      |
| EfficientNetB3+cutmix  |  0.958    |      |      |
| EfficientNetB3+snapmix |  1.0      |      |      |

## References
* [https://github.com/Shaoli-Huang/SnapMix](https://github.com/Shaoli-Huang/SnapMix)
* [SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data](https://arxiv.org/abs/2012.04846)
