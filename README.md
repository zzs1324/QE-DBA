# BO-DBA
BO-DBA: Efficient Decision-Based Adversarial Attacks via Bayesian Optimization
## Setup

Install the required libraries:
```
pip install -r requirements.txt 
```
Download ImageNet test data: [ILSVRC2012_img_val.tar](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5), and put them under folder `BO-DBA/DataSet/`

Data Preparation
```
cd DataSet
Python preprocess_imagenet_validation_data.py
```
