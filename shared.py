
import cv2
import random
import GPyOpt as gy
#import noise as ns
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import noise as ns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from scipy import ndimage

#tf.test.is_gpu_available()

main_path = './' #path of main folder
data_path = os.path.join(main_path, "DataSet/ILSVRC2012_img_val") #path of validation data

images = {}
classfiles = os.listdir(data_path)

for clf in classfiles:
    images[clf] = os.listdir(os.path.join(data_path,clf));
mod = "ResNet"
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
    
# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
    return decode_predictions(probs, top=6)[0]


def display_images(image):
    guessdata = get_imagenet_label(pretrained_model.predict(image, steps=1))
    for guess in guessdata:
        print(guess[1] + ": " + str(guess[2]))
        
    plt.figure()
    plt.imshow(image[0])
    plt.show()
    
def importimage(imgpath):
    rawimage = Image.open(imgpath)
    image = tf.keras.preprocessing.image.img_to_array(rawimage)

    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.image.resize(image, (imagesize, imagesize))
    image = image[None, ...]

    if np.shape(image) == (1,imagesize,imagesize,1):
        image = tf.image.grayscale_to_rgb(image)
    elif np.shape(image) == (1,imagesize,imagesize,4):
        image = tf.image.grayscale_to_rgb(image)

    return rawimage, image
    
class randomimg:
    
    def __init__(self, m="joint", t=-1):
        cls = random.choice(classfiles)
        imgfile = random.choice(images[cls])
        imgpath = os.path.join(data_path, cls+"/"+imgfile)      
            
        rawimage, image = importimage(imgpath)
                    
        self.q = 0
        
        self.imgplot = rawimage 
        self.method = m
        self.threshold = t
 
            
        #print(np.shape(image))
        self.img = image
        self.image_probs = get_imagenet_label(pretrained_model.predict(image, steps=1))
        self.labelindex = np.argmax(pretrained_model.predict(image, steps=1))
        origpredictions = self.image_probs[0]
    
        actualprediction = os.path.basename(os.path.dirname(imgpath))
        

        if origpredictions[0] != actualprediction:
            self = randomimg()

    # def decision(self,img):#This one is Detection free
    #     check = get_imagenet_label(pretrained_model.predict(img, steps=1))
    #     self.q += 1
    #     if self.threshold == -1:
    #         check2 = adversarial_detection(img, self.method)[0]
    #     else:
    #         check2 = adversarial_detection(img, self.method, self.threshold)[0]
    #     if (check[0][0] == self.image_probs[0][0]) or not (check2):
    #         return False
    #     else:
    #         return True

    def decision(self,img):#This one include detection algorithm
        check = get_imagenet_label(pretrained_model.predict(img, steps=1))
        self.q += 1
        if self.threshold == -1:
            check2 = adversarial_detection(img, self.method)[0]#True means detected
        else:
            check2 = adversarial_detection(img, self.method, self.threshold)[0]
        if (check[0][0] == self.image_probs[0][0]) or check2:
            return False#attack fail(label did not change or been detected)
        else:
            return True

l1_dist = lambda x1,x2: np.sum(np.abs(x1 - x2), axis=tuple(range(len(x1.shape))[1:]))

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
    return ndimage.filters.median_filter(x, size=(1,1,width,height), mode='reflect')
                                         
                                         
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
    ret_img = cv2.fastNlMeansDenoisingColored(i, None,a,a,b,c)
    ret_img = np.expand_dims(ret_img, 0)
    ret_img = tf.convert_to_tensor(ret_img)
    return ret_img/255

#threshold = 1.2128
#optimal imagenet threshold from paper

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
        squeezed = non_local_mean(im,11,3,4)
    if method == "joint":
        dist1 = adversarial_detection(im, "bit_depth")[1]
        dist2 = adversarial_detection(im, "median_smoothing")[1]
        dist3 = adversarial_detection(im, "non_local_mean")[1]
        preddist = max([dist1, dist2, dist3])
        #print(preddist, threshold)
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
    cutoff = int(np.round(len(imglist)*(percentile/100))-1)
    distlist = np.sort(distlist)
    print(distlist)
    return distlist[cutoff]


    
def norm (image, image2):
    if isinstance(image, np.ndarray):
        y = image
    else:
        y = image.numpy()[0]
    if isinstance(image2, np.ndarray):
        z = image2
    else:
        z = image2.numpy()[0]
    l2norm = tf.norm(np.subtract(z,y), ord=2).numpy()
    infnorm = tf.norm(np.subtract(z,y), ord=np.inf).numpy()
    return l2norm, infnorm  


def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)    

def project(original_image, perturbed_image, alphas, l):
    
    #alphas_shape = len(original_image.shape)
    #alphas = alphas.reshape(alphas_shape)
    if l == 'l2':
        return (1-alphas) * original_image + alphas * perturbed_image
    elif l == 'linf':
        out_images = clip_image(
            perturbed_image, 
            original_image - alphas, 
            original_image + alphas
        )
        return out_images
    
def binary_search(perturbed_image, imgobj, theta, l='l2'):
    low = 0

    while imgobj.decision(perturbed_image)==0:
        perturbed_image1 = perturbed_image
        perturbed_image = (perturbed_image-imgobj.img)*2+imgobj.img
        perturbed_image = np.clip(perturbed_image,0,1)
        if np.array_equal(perturbed_image1,perturbed_image):
            print('inf happened')
            return perturbed_image, float('inf')
#        display_images(perturbed_image)
        low = 0.5
#        print(imgobj.q,end=",")

    if l == 'l2':
        distance= norm(perturbed_image, imgobj.img)[0]
        high = 1
    else:
        distance= norm(perturbed_image, imgobj.img)[1]
        high = distance
            


    
    while (high - low) / theta > 1:
#        print(imgobj.q,end=";")
        mid = (high + low) / 2.0
        mid_image = project(imgobj.img, perturbed_image, mid, l)
        d = imgobj.decision(mid_image)
        if d ==1:
            high = mid
        else:
            low = mid

    output = project(imgobj.img, perturbed_image, high, l)

    if l == 'l2':
        finaldist = norm(output, imgobj.img)[0]
    else:
        finaldist = norm(output, imgobj.img)[1]
            
    #print(theta)
    if l=='l2':
        out_image = output.numpy()
    else:
        out_image = output
    return out_image, finaldist

def select_delta(dist, l, cur_iter, theta, d):
    if cur_iter == 1:
        delta = 0.1
    else:
        if l == 'l2':
            delta=np.sqrt(d)*theta*dist
        elif l == 'linf':
            delta=np.sqrt(d)*theta*dist
    return delta
def perlin_noise(noise_scale, noise_octaves,  color_freq, noise_p=1, noise_l=2):
    blank = np.zeros((imagesize,imagesize, 3))
    for i in range(imagesize):
        for j in range(imagesize):
            for k in range(3):
                blank[i][j][k] = .5+ns.pnoise2(i/int(noise_scale), 
                                      j/int(noise_scale), 
                                      octaves=int(noise_octaves), 
                                      persistence=float(noise_p), 
                                      lacunarity=float(noise_l), 
                                      repeatx=imagesize, 
                                      repeaty=imagesize, 
                                      base = 0
                                      )
    blank = np.sin(blank*color_freq*np.pi)
      
    return normalize(blank)
def normalize(vec):
    vmax = np.amax(vec)
    vmin  = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)

def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True

def gaborK(ksize, sigma, theta, lambd, xy_ratio, sides):
    gabor_kern = cv2.getGaborKernel((int(ksize), int(ksize)), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    for i in range(1, int(sides)):
        gabor_kern += cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi * i / sides, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    return gabor_kern

def gabor_noise_random(num_kern, ksize, sigma, theta, lambd, xy_ratio = 1, sides= 1, seed = 0):
    
    grid = 20
    np.random.seed(seed)
    
    # Gabor kernel
    if sides != 1: gabor_kern = gaborK(ksize, sigma, theta, lambd, xy_ratio, sides)
    else: gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype = cv2.CV_32F)
    
    # Sparse convolution noise
    sp_conv = np.zeros([imagesize, imagesize])
    dim = int(imagesize / 2 // grid)
    noise = []
    for i in range(-dim, dim + 1):
        for j in range(-dim, dim + 1):
            x = i * grid + imagesize / 2 - grid / 2
            y = j * grid + imagesize / 2 - grid / 2
            for _ in range(num_kern):
                dx = np.random.randint(0, grid)
                dy = np.random.randint(0, grid)
                while not valid_position(imagesize, x + dx, y + dy):
                    dx = np.random.randint(0, grid)
                    dy = np.random.randint(0, grid)
                weight = np.random.random() * 2 - 1
                sp_conv[int(x + dx)][int(y + dy)] = weight
                
    sp_conv = cv2.filter2D(sp_conv, -1, gabor_kern)
    
    normn = normalize(sp_conv)
    normn = np.around(normn)
    normn = np.expand_dims(normn, 0)
    normn = np.expand_dims(normn, 3)
    normn = tf.image.grayscale_to_rgb(tf.convert_to_tensor(normn))
    
    return normn.numpy()


import numpy as np
import colorsys

rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = (h+hout)%1
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b))
    return arr

def colorize(image, hue):
    
    new_img = shift_hue(image, hue)

    return new_img

import pickle
def SAVE(fp,input):
    with open(fp, "wb+") as fp:
        pickle.dump(input, fp)
    return


def LOAD(fp):
    with open(fp, "rb+") as fp:
        output = pickle.load(fp)
    return output

import time
def millis():
    return int(round(time.time() * 1000))