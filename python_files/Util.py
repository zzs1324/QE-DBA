import cv2
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from python_files.shared import norm
import os
from scipy import ndimage
import yaml
import time
import keras
import pickle

def SAVE(fp, input):
    with open(fp, "wb+") as fp:
        pickle.dump(input, fp)
    return


def LOAD(fp):
    with open(fp, "rb+") as fp:
        output = pickle.load(fp)
    return output

def millis():
    return int(round(time.time() * 1000))

#Load data and model
with open(f'./Configuration.yaml', 'r') as f:
    params = yaml.safe_load(f)

main_path = params['main_path']
data_path = params['data_path']

images = {}
classfiles = os.listdir(data_path)
cls = random.choice(classfiles)

for clf in classfiles:
    images[clf] = os.listdir(os.path.join(data_path, clf));
mod = params['mod']
if mod == "ResNet":
    pretrained_model = tf.keras.applications.ResNet50V2(
        weights='imagenet')
    pretrained_model.trainable = False
    decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
    imagesize = 224

elif mod == "Inception":
    pretrained_model = tf.keras.applications.InceptionV3(
        weights='imagenet')
    pretrained_model.trainable = False
    decode_predictions = tf.keras.applications.inception_v3.decode_predictions
    imagesize = 299
elif mod =="MNIST":
    pretrained_model = keras.models.load_model('./Models/Mnist')
    imagesize=28
    num_classes=10
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255
    x_test = np.expand_dims(x_test, -1)
    y_test = keras.utils.to_categorical(y_test, num_classes)
elif mod =="CIFAR":
    pretrained_model = keras.models.load_model('./Models/CIFAR10')
    imagesize=32
    num_classes=10
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255
    y_test = keras.utils.to_categorical(y_test, num_classes)



def norm(image, image2):
    if isinstance(image, np.ndarray):
        y = image
    else:
        y = image.numpy()[0]
    if isinstance(image2, np.ndarray):
        z = image2
    else:
        z = image2.numpy()[0]
    l2norm = tf.norm(np.subtract(z, y), ord=2).numpy()
    infnorm = tf.norm(np.subtract(z, y), ord=np.inf).numpy()
    return l2norm, infnorm

def get_imagenet_label(probs):
    if mod=="MNIST" or mod=="CIFAR":
        res = []
        probs=probs[0]
        idxlist = np.argsort(-probs)
        #print(probs)
        #print(idxlist)
        for i in range(6):
            res.append((idxlist[i],str(idxlist[i]),probs[idxlist[i]]))
        return res
    else:
        return decode_predictions(probs, top=6)[0]


def display_images(image):
    guessdata = get_imagenet_label(pretrained_model.predict(image, steps=1))
    for guess in guessdata:
        print(guess[1] + ": " + str(guess[2]))
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image[0])
    plt.show()

def importimage(imgpath):
    rawimage = Image.open(imgpath)
    image = tf.keras.preprocessing.image.img_to_array(rawimage)

    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (imagesize, imagesize))
    image = image[None, ...]

    if np.shape(image) == (1, imagesize, imagesize, 1):
        image = tf.image.grayscale_to_rgb(image)
    elif np.shape(image) == (1, imagesize, imagesize, 4):
        image = tf.image.grayscale_to_rgb(image)

    return rawimage, image


class randomimg:

    def __init__(self, m="joint", t=-1,mode='Raw',target=0):
        # recorder distance
        self.maxl2 = float('inf')
        self.maxlinf = float('inf')
        self.historyl2 = []
        self.historylinf = []
        self.q = 0
        self.method = m
        self.threshold = t
        self.mode = mode
        self.target = target
        

        if mod =="MNIST":
            index = random.choice(range(len(x_test)))
            image = np.expand_dims(x_test[index],0)
            rawimage = x_test[index]
            self.img = tf.convert_to_tensor(image, dtype=tf.float32)
            self.image_probs = get_imagenet_label(pretrained_model.predict(image, steps=1))
            self.labelindex = np.argmax(pretrained_model.predict(image, steps=1))
            origpredictions = self.image_probs[0]
            #print(origpredictions)
            actualprediction = np.argmax(y_test)
            #print(actualprediction)
        elif mod=="CIFAR":
            index = random.choice(range(len(x_test)))
            image = np.expand_dims(x_test[index],0)
            rawimage = x_test[index]
            self.img = tf.convert_to_tensor(image, dtype=tf.float32)
            self.image_probs = get_imagenet_label(pretrained_model.predict(image, steps=1))
            self.labelindex = np.argmax(pretrained_model.predict(image, steps=1))
            origpredictions = self.image_probs[0]
            #print(origpredictions)
            actualprediction = np.argmax(y_test)
        else:
            cls = random.choice(classfiles)
            imgfile = random.choice(images[cls])
            imgpath = os.path.join(data_path, cls + "/" + imgfile)

            rawimage, image = importimage(imgpath)
            # print(np.shape(image))
            self.img = image
            self.image_probs = get_imagenet_label(pretrained_model.predict(image, steps=1))
            self.labelindex = np.argmax(pretrained_model.predict(image, steps=1))
            origpredictions = self.image_probs[0]
            actualprediction = os.path.basename(os.path.dirname(imgpath))


        if origpredictions[0] != actualprediction:
            self = randomimg()


    def update(self,img):
        twonorm,infnorm = norm(self.img, img)
        if twonorm<self.maxl2:
            self.maxl2 = twonorm
            self.historyl2.append([self.q,twonorm])
        if infnorm<self.maxlinf:
            self.maxlinf = infnorm
            self.historylinf.append([self.q,infnorm])

    def decision(self, img):
        check = get_imagenet_label(pretrained_model.predict(img, steps=1))
        self.q += 1
        if self.mode=='Target':
            if check[0][0]==self.target:
                result=True
            else:
                result=False
        else:
            if check[0][0] == self.image_probs[0][0]:
                result = False
            else:
                result = True
        if self.mode=='Detection':#run detection
            if self.threshold == -1:
                detection = adversarial_detection(img, self.method)[0]  # True means detected
            else:
                detection = adversarial_detection(img, self.method, self.threshold)[0]
            if detection==True:
                result = False  # attack fail(label did not change or been detected)
        if result==True:
            self.update(img)
        return result

#DefenseMethod
l1_dist = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))


def median_smoothing(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=(1, 1, width, height), mode='reflect')


def bit_depth(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    if tf.is_tensor(x):
        x = x.numpy()
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float


def non_local_mean(x, a, b, c):
    if tf.is_tensor(x):
        x = x.numpy()
    i = x[0]
    i = cv2.normalize(i, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    ret_img = cv2.fastNlMeansDenoisingColored(i, None, a, a, b, c)
    ret_img = np.expand_dims(ret_img, 0)
    ret_img = tf.convert_to_tensor(ret_img)
    return ret_img / 255


# threshold = 1.2128
# optimal imagenet threshold from paper

def adversarial_detection(im, method, threshold=-1):
    if threshold == -1:
        if method == "bit_depth":
            threshold = .307
        elif method == "median_smoothing":
            threshold = .940
        elif method == "non_local_mean":
            threshold = .623
        elif method == "joint":
            threshold = 1.307

    originalpred = pretrained_model.predict(im, steps=1)
    if method == "bit_depth":
        squeezed = bit_depth(im, 32)
    elif method == "median_smoothing":
        squeezed = median_smoothing(im, 2)
    elif method == "non_local_mean":
        squeezed = non_local_mean(im, 11, 3, 4)
    if method == "joint":
        dist1 = adversarial_detection(im, "bit_depth")[1]
        dist2 = adversarial_detection(im, "median_smoothing")[1]
        dist3 = adversarial_detection(im, "non_local_mean")[1]
        preddist = max([dist1, dist2, dist3])
        # print(preddist, threshold)
    else:
        newpred = pretrained_model.predict(squeezed, steps=1)
        preddist = l1_dist(np.array([originalpred]), np.array([newpred]))
    if preddist > threshold:
        return True, preddist
    else:
        return False, preddist


def getthreshold(imglist, dettype, percentile):
    distlist = []
    for orig in imglist:
        origdetect = adversarial_detection(orig, dettype)
        distlist.append(origdetect[1][0])

    print(np.percentile(distlist, percentile))
    cutoff = int(np.round(len(imglist) * (percentile / 100)) - 1)
    distlist = np.sort(distlist)
    print(distlist)
    return distlist[cutoff]

def ResultSave(Name,Path):
    FileName = Path+'/'+Name+'.dat'
    DirPath = Path+'/'+Name
    if not os.path.isdir(DirPath):
        os.mkdir(DirPath)
    ImgPrefix = DirPath+'/'+Name+'_'
    return FileName,ImgPrefix


def DemoVisulization(oriImg, Adversary, History, queryBudgets, fontsize=20, SavePath=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))

    ax1.imshow(oriImg[0])
    ax1.set_title('Original Image', size=fontsize)
    ax1.set_axis_off()
    guessdata = get_imagenet_label(pretrained_model.predict(oriImg, steps=1))
    predict = ''
    for guess in guessdata:
        predict = predict + guess[1] + ": " + '{:.3f}'.format(guess[2]) + '\n'
    ax1.text(1, -0.01, predict, ha="right", va='top', size=fontsize * 0.8, transform=ax1.transAxes)

    ax2.imshow(Adversary[0])
    ax2.set_title('Adversarial Example', size=fontsize)
    ax2.set_axis_off()
    guessdata = get_imagenet_label(pretrained_model.predict(Adversary, steps=1))
    predict = ''
    for guess in guessdata:
        predict = predict + guess[1] + ": " + '{:.3f}'.format(guess[2]) + '\n'
    ax2.text(1, -0.01, predict, ha="right", va='top', size=fontsize * 0.8, transform=ax2.transAxes)
    perturbation = (Adversary[0] - oriImg[0])*255
    ax3.imshow(perturbation.numpy().astype('int'))
    ax3.set_title('Pertubation', size=fontsize)
    ax3.set_axis_off()

    fig.tight_layout()
    if SavePath != None:
        plt.savefig(SavePath, bbox_inches='tight', pad_inches=0)

    x = range(queryBudgets)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    X1 = [i[0] for i in History[0]]
    Y1 = [i[1] for i in History[0]]
    Y1 = np.interp(x, X1, Y1)
    ax1.plot(x, Y1, '-', color='r', linestyle='-')
    ax1.set_title('$l_2$ Distance', size=fontsize)
    # ax1.set_ylabel('$l_2$ Distance',size=fontsize)

    X1 = [i[0] for i in History[1]]
    Y1 = [i[1] for i in History[1]]
    Y1 = np.interp(x, X1, Y1)
    ax2.plot(x, Y1, '-', color='r', linestyle='-')
    ax2.set_title('$l_\infty$ Distance', size=fontsize)

    X1 = [i[0] for i in History[2]]
    Y1 = [i[1] / 1000 for i in History[2]]
    Y1 = np.interp(x, X1, Y1)
    ax3.plot(x, Y1, '-', color='r', linestyle='-')
    ax3.set_title('Time(s)', size=fontsize)
