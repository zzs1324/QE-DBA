# BO-DBA (Bayesian Optimization Decision Based Adversarial Attacks) ✏️

### Here is a brief introduction of what this project is about:
* BO-DBA is - Query-Efficient Decision-Based Adversarial Attacks via Bayesian Optimization. 
* Whenever Adversarial examples are concerned, the goal is to always minimize the distance between the original image and the perturbated image, subject to the constraint that we use the same model on both images.
* So in this project we are aiming to do so by using Bayesian Optimization.
* The result should be categorising the two images, different from each other, even if they look alike to the human eye.


## Now getting back to the project implementation-

### ⚙️ Setup

**Step 1: Install the required libraries:**

For this particular project there are many things that are required to be installed before you run the program. All of these are given in the 'requirements.txt' file. Just give that file a glance to know what all modules you should install. However, if you feel like installing every single module mentioned in that file then just simply copy, paste and run the command given below using 'pip'. *As simple as it looks like* 💁‍♂️.
```
pip install -r requirements.txt 
```
**Step 2: Accessing Dataset**

* You can download the ImageNet test dataset required for this project from here ➡️[ILSVRC2012_img_val.tar](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5). 
* Then you have to put this compressed folder inside the `BO-DBA/DataSet/` folder.
* Now, all you need to do is to decompress it.

**Step 3: Data Preparation**

For this step, first set your directory to DataSet and then run the 'Python preprocess_imagenet_validation_data.py' file. The commands that you have to type are given below:
```
cd DataSet
Python preprocess_imagenet_validation_data.py
```
## 📄Configuration
In Configuration.yaml, set mod "Inception" or "ResNet" to choose targeted classifier.

For detail configuration, see `BO-DBA/Demo.ipynb`.

## 💻Running
To see the correct implementation of the project, first have a look at ➡️  `BO-DBA/Demo.ipynb`. This jupyter notebook implements the BO-DBA attack and makes it easier to understand. The .ipynb file also provides the comparision of different types of adversarial attacks. Thus, explaining how our approach of applying Bayesian Optimization is better than the others. 

If you wish to see the source code of all the evaluation experiments then please have a look at this -> `BO-DBA/Evaluation/`. Just remember to copy the codes to the main folder before running it!
## 💡Authors

- [@prashantjadiya](https://github.com/prashantjadiya)
- [@zzs1324](https://github.com/zzs1324)
- [@nmcdermo](https://github.com/nmcdermo)

