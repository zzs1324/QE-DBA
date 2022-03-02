#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
INSTRUCTIONS BELOW
"""
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray

import os
import numpy as np
import lpips
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
"""
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
"""

"""
folder = "../Result/AdversaryExample/"
im1name= "Original_24.npy"
im2name = "Ver3_24.npy"
im1 = np.load(os.path.join(folder,im1name))
im2 = np.load(os.path.join(folder,im2name))
"""

preload = lpips.LPIPS(net='alex')

def special_distance(im1, im2, mode):
    if mode == "lpips":
        im1 = torch.tensor(np.transpose(im1, [0,3,1,2]))
        im2 = torch.tensor(np.transpose(im2, [0,3,1,2]))
        return preload(im1, im2).detach().numpy().item()
    elif mode == "ssim" or mode == "msssim" or mode == "psnr":
        im1 =tf.convert_to_tensor(im1[0])
        im2 = tf.convert_to_tensor(im2[0])
        if mode == "ssim":
            return tf.image.ssim(im1, im2, max_val=1.0).numpy(  )
        elif mode == "msssim":
            return tf.image.ssim_multiscale(im1, im2, max_val=1.0).numpy()
        elif mode == "psnr":
            return tf.image.psnr(im1, im2, max_val=1.0).numpy()
    elif mode == "entropy":
        diff = rgb2gray(im1)-rgb2gray(im2)
        return shannon_entropy(diff)

def comparesavedimages(methodname, distancemode, rng):
    reslist = []
    for i in rng:
        orname = "Original_" + str(i) + ".npy"
        advname = methodname + "_" + str(i) + ".npy"
        adv = None
        ori = np.load(os.path.join(folder, orname))
        try:
            adv = np.load(os.path.join(folder, advname))
        except:
            pass
            #print(methodname + " " + str(i) + " missing")
        finally:
            reslist.append(special_distance(ori, adv, distancemode))
    return reslist

def ignoremissing():
    custrange = []
    for i in range(100):
        if os.path.exists(os.path.join(folder,"Procedural_" + str(i) + ".npy")):
            custrange.append(i)
    return custrange

def histo(methodnames, distancemode, ignorefailedprocedural=True):
    ptitle = distancemode + " Histogram"
    histdata = []
    if ignorefailedprocedural:
        rang = ignoremissing()
    else:
        rang = range(100)

    for method in methodnames:
        histdata.append(comparesavedimages(method, distancemode, rang))
        
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title(ptitle)   
    plt.hist(histdata)
    plt.legend(methodnames)

    plt.show()

def averages(types, ignorefailedprocedural=True):
    methods = ["lpips","ssim", "psnr", "msssim", "entropy"]
    colors = ["b", "g", "r", "y", "m"]
    if ignorefailedprocedural:
        rang = ignoremissing()
    else:
        rang = range(100)   
    f=plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    fig, ax = plt.subplots(1,5, figsize=(20, 8))

    for i in range(len(methods)):
        avgs = []
        for t in types:
            avgs.append(np.mean(comparesavedimages( t,methods[i], rang)))
      
        ax[i].bar( types, avgs, color=colors[i]*5)
        ax[i].title.set_text((methods[i]+" Average"))

"""
def histo: gives a histogram of how the various attacks perform on a specific distance method"
Argument 1: A list of the types.  Should be in the name of the image files
ex "Ver3 -> Ver3_81.npy"
Argument 2: Distance method.  Can be "lpips", "ssim", "psnr", "msssim", "entropy"
Argument 3: Bool, Ignore images where procedural noise failed
"""

"""
def averages: gives a histogram of how the various attacks perform on a specific distance method"
Argument 1: A list of the types.  Should be in the name of the image files
ex "Ver3 -> Ver3_81.npy"
Argument 2: Bool, Ignore images where procedural noise failed
"""

"""
types = ["Procedural","Ver3", "Ver4", "Ver4_12_HUE"]
averages(types)
histo(types, "lpips")
histo(types, "ssim")
histo(types, "psnr")
histo(types, "msssim")
histo(types, "entropy")
"""
