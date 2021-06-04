# BO-DBA
BO-DBA: Query-Efficient Decision-Based Adversarial Attacks via Bayesian Optimization
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
## Configuration
In Configuration.yaml, set mod "Inception" or "ResNet" to choose targeted classifier.

For detail configuration, see `BO-DBA/Demo.ipynb`.

## Running
We provide `BO-DBA/Demo.ipynb` as an demo to run BO-DBA attack and other attacks we compared with in our paper.

We also provide source code of evaluation experiments in `BO-DBA/Evaluation/`, copy the codes to main folder before running it.
