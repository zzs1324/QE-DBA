# BO-DBA
BO-DBA: Efficient Decision-Based Adversarial Attacks via Bayesian Optimization
## Setup

Install the required libraries:
```
pip install -r requirements.txt 
```
Download ImageNet test data: [ImageNet test images](http://www-personal.umich.edu/~timtu/Downloads/imagenet_npy/imagenet_test_data.npy) and [ImageNet test labels](http://www-personal.umich.edu/~timtu/Downloads/imagenet_npy/imagenet_test_labels.npy) , and put them under folder `BO-DBA/DataSet/`

Data Preparation
```
cd DataSet
Python preprocess_imagenet_validation_data.py
```
