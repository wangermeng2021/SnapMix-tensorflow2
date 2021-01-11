

# Installation
##  1. Install environment
* Conda(recommend)<br>

  This might be the simplest way to install tensorflow with cuda:
  ```
  conda env create -f conda-gpu.yml
  conda activate snapmix-tensorflow2
  ```
* Pip
  ```
  pip install -r requirements.txt
  ```
	PS: If you use pip installation，you may need to install cuda11 if you have not installed it,Because tensorflow2.4 require cuda11.

##   2. Clone project
```
git clone https://github.com/wangermeng2021/snapmix-tensorflow2
cd snapmix-tensorflow2
```

##   3. Download dataset
* Download cub dataset
  ```
  wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz -P dataset/cars/
  tar -xzf CUB_200_2011.tgz -C dataset/cub
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

# Training
* For training on cub dataset,use:
  ```
  python train.py --dataset cub
  ```
* For training on Cars dataset,use:
  ```
  python train.py --dataset car
  ```

# References
* [SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data](https://arxiv.org/abs/2012.04846)
