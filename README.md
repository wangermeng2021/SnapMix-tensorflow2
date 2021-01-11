
# SnapMix-tensorflow2
tensorflow2 implementation of SnapMix as described in [SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data](https://arxiv.org/abs/2012.04846)

## Installation
###  1. Install environment
  ```
  pip install -r requirements.txt
  ```

###   2. Clone project
```
git clone https://github.com/wangermeng2021/SnapMix-tensorflow2
cd SnapMix-tensorflow2
```

###   3. Download dataset
* Download cub dataset
  ```
  wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz -P dataset/cub/
  tar -xzf dataset/cub/CUB_200_2011.tgz -C dataset/cub
  rm dataset/cub/CUB_200_2011.tgz
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
  rm dataset/cars/cars_train.tgz
  rm dataset/cars/cars_test.tgz
  rm dataset/cars/car_devkit.tgz
  ```

## Training
* For training on cub dataset,use:
  ```
  python train.py --dataset cub
  ```
* For training on Cars dataset,use:
  ```
  python train.py --dataset car
  ```

## References
* [SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data](https://arxiv.org/abs/2012.04846)
