import pickle
import time
import numpy as np
import colorsys
import tensorflow as tf
from Util import imagesize
import GPyOpt as gy
import noise as ns
import cv2
from tqdm.notebook import tqdm

def SAVE(fp,input):
    with open(fp, "wb+") as fp:
        pickle.dump(input, fp)
    return
def LOAD(fp):
    with open(fp, "rb+") as fp:
        output = pickle.load(fp)
    return output
def millis():
    return int(round(time.time() * 1000))


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

def binary_search(perturbed_image, imgobj, theta, l='l2'):
    low = 0

    while imgobj.decision(perturbed_image) == 0:
        perturbed_image1 = perturbed_image
        perturbed_image = (perturbed_image - imgobj.img) * 2 + imgobj.img
        perturbed_image = np.clip(perturbed_image, 0, 1)
        if np.array_equal(perturbed_image1, perturbed_image):
            print('inf happened')
            return perturbed_image, float('inf')
        #        display_images(perturbed_image)
        low = 0.5
    #        print(imgobj.q,end=",")

    if l == 'l2':
        distance = norm(perturbed_image, imgobj.img)[0]
        high = 1
    else:
        distance = norm(perturbed_image, imgobj.img)[1]
        high = distance

    while (high - low) / theta > 1:
        #        print(imgobj.q,end=";")
        mid = (high + low) / 2.0
        mid_image = project(imgobj.img, perturbed_image, mid, l)
        d = imgobj.decision(mid_image)
        if d == 1:
            high = mid
        else:
            low = mid

    output = project(imgobj.img, perturbed_image, high, l)

    if l == 'l2':
        finaldist = norm(output, imgobj.img)[0]
    else:
        finaldist = norm(output, imgobj.img)[1]

    # print(theta)
    if l == 'l2':
        out_image = output.numpy()
    else:
        out_image = output
    return out_image, finaldist


def select_delta(dist, l, cur_iter, theta, d):
    if cur_iter == 1:
        delta = 0.1
    else:
        if l == 'l2':
            delta = np.sqrt(d) * theta * dist
        elif l == 'linf':
            delta = np.sqrt(d) * theta * dist
    return delta

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)


def project(original_image, perturbed_image, alphas, l):
    # alphas_shape = len(original_image.shape)
    # alphas = alphas.reshape(alphas_shape)
    if l == 'l2':
        return (1 - alphas) * original_image + alphas * perturbed_image
    elif l == 'linf':
        out_images = clip_image(
            perturbed_image,
            original_image - alphas,
            original_image + alphas
        )
        return out_images

def perlin_noise(noise_scale, noise_octaves, color_freq, noise_p=1, noise_l=2):
    blank = np.zeros((imagesize, imagesize, 3))
    for i in range(imagesize):
        for j in range(imagesize):
            for k in range(3):
                blank[i][j][k] = .5 + ns.pnoise2(i / int(noise_scale),
                                                 j / int(noise_scale),
                                                 octaves=int(noise_octaves),
                                                 persistence=float(noise_p),
                                                 lacunarity=float(noise_l),
                                                 repeatx=imagesize,
                                                 repeaty=imagesize,
                                                 base=0
                                                 )
    blank = np.sin(blank * color_freq * np.pi)

    return normalize(blank)


def normalize(vec):
    vmax = np.amax(vec)
    vmin = np.amin(vec)
    return (vec - vmin) / (vmax - vmin)


def valid_position(size, x, y):
    if x < 0 or x >= size: return False
    if y < 0 or y >= size: return False
    return True


def gaborK(ksize, sigma, theta, lambd, xy_ratio, sides):
    gabor_kern = cv2.getGaborKernel((int(ksize), int(ksize)), sigma, theta, lambd, xy_ratio, 0, ktype=cv2.CV_32F)
    for i in range(1, int(sides)):
        gabor_kern += cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi * i / sides, lambd, xy_ratio, 0,
                                         ktype=cv2.CV_32F)
    return gabor_kern


def gabor_noise_random(num_kern, ksize, sigma, theta, lambd, xy_ratio=1, sides=1, seed=0):
    grid = 20
    np.random.seed(seed)

    # Gabor kernel
    if sides != 1:
        gabor_kern = gaborK(ksize, sigma, theta, lambd, xy_ratio, sides)
    else:
        gabor_kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, xy_ratio, 0, ktype=cv2.CV_32F)

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




rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = (h + hout) % 1
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b))
    return arr

def colorize(image, hue):
    new_img = shift_hue(image, hue)

    return new_img
from Upsample import upsample_projection
def create_distorted_image(image, typ, epsilon, parameters):
    parameters = parameters[0]
    if typ == 'perlin':
        pert = perlin_noise(parameters[0], parameters[1], parameters[2])
    elif typ == 'gabor':
        pert = gabor_noise_random(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3], parameters[4], sides = int(parameters[5]))
    elif typ =='BICU' or typ == 'CLUSTER' or typ =='NN' or typ =='BILI':
        parameters = np.expand_dims(parameters, axis=0)
        pert = upsample_projection(typ, parameters, 16, 224 * 224, nchannel=3)
        pert = pert.reshape(1, 3, 224, 224)
        pert = pert.transpose(0, 2, 3, 1)
        pert = (pert - np.min(pert)) / np.ptp(pert)

    pert = (pert-.5)*2

    dist_img = image + epsilon*pert
    return dist_img

class preddifference:
    def __init__(self, image, maxnorm, noise, constraint='l2'):
        self.image = image
        self.maxnorm = maxnorm
        self.noise = noise
        self.best = float('inf')
        self.theta = 0.01
        self.constraint=constraint
        pass

    def func(self, parameters):
        final = create_distorted_image(self.image.img, self.noise, self.maxnorm / 255, parameters)

        out, dist = binary_search(final, self.image, self.theta, l=self.constraint)
        if dist < self.best:
            self.best = dist
            self.adv = out
        return dist #np.log(dist*(255/self.maxnorm))

def bayesian_attack(image, max_query, init_query=5, noise='perlin', max_norm=16,constraint='l2'):
    if noise == 'perlin':
        bounds = [{'name': 'wavelength', 'type': 'continuous', 'domain': (10, 200), 'dimensionality': 1},
                  {'name': 'octave', 'type': 'discrete', 'domain': (1, 2, 3, 4), 'dimensionality': 1},
                  {'name': 'freq_sine', 'type': 'continuous', 'domain': (4, 32), 'dimensionality': 1}
                  ]
    elif noise == 'gabor':
        bounds = [{'name': 'kernels', 'type': 'discrete', 'domain': (1, 200), 'dimensionality': 1},
                  {'name': 'kernel size', 'type': 'discrete', 'domain': (1, 40), 'dimensionality': 1},
                  {'name': 'sigma', 'type': 'continuous', 'domain': (1, 8), 'dimensionality': 1},
                  {'name': 'orientation', 'type': 'continuous', 'domain': (0, 2 * np.pi), 'dimensionality': 1},
                  {'name': 'scale', 'type': 'continuous', 'domain': (1, 20), 'dimensionality': 1},
                  {'name': 'sides', 'type': 'discrete', 'domain': (1, 12), 'dimensionality': 1}
                  ]
    else:
        bounds = [{'name': 'wavelength', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 48}]

    feasible_space = gy.Design_space(space=bounds)
    initial_design = gy.experiment_design.initial_design('random', feasible_space, init_query)

    queries = 0


    optimized = preddifference(image, max_norm, noise=noise,constraint=constraint)

    # Gaussian process and Bayesian optimization
    objective = gy.core.task.SingleObjective(optimized.func, num_cores=1)
    model = gy.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
    aquisition_opt = gy.optimization.AcquisitionOptimizer(feasible_space)
    acquisition = gy.acquisitions.AcquisitionLCB(model, feasible_space, optimizer=aquisition_opt)
    evaluator = gy.core.evaluators.Sequential(acquisition, batch_size=1)
    BOpt = gy.methods.ModularBayesianOptimization(model, feasible_space, objective, acquisition, evaluator,
                                                  initial_design)
    HistoryL2 = []
    HistoryLinf = []
    bestDistanceL2 = 2550
    bestDistanceLinf = 2550
    best_f = float('inf')
    TimeHistory = [[0, 0]]
    t1 = millis()
    success = False
    last = 0
    t = tqdm(total=max_query)
    while image.q < max_query:
        BOpt.run_optimization(max_iter=1)
        t.n=image.q
        t.update(n=0)
        if image.q == last:
            break
        else:
            last = image.q
        t2 = millis()
        TimeHistory.append([image.q, t2 - t1])
        if BOpt.fx_opt < best_f:
            best_f = BOpt.fx_opt
            twonorm, infnorm = norm(image.img, optimized.adv)
            if infnorm < bestDistanceLinf:
                bestDistanceLinf = infnorm
                HistoryLinf.append([image.q, bestDistanceLinf])
                if constraint=='linf':
                    success=True
                    final = optimized.adv
            if twonorm < bestDistanceL2:
                bestDistanceL2 = twonorm
                HistoryL2.append([image.q, bestDistanceL2])
                if constraint=='l2':
                    success=True
                    final = optimized.adv
    HistoryL2.append([image.q, bestDistanceL2])
    HistoryLinf.append([image.q, bestDistanceLinf])
    twonorm, infnorm = norm(image.img, final)
    return success,final, [HistoryL2, HistoryLinf, TimeHistory],[twonorm,infnorm]

#QueryAttack
import random
import os
from Util import classfiles,images,data_path,importimage
def get_Random_Img():
    cls = random.choice(classfiles)
    imgfile = random.choice(images[cls])
    imgpath = os.path.join(data_path, cls + "/" + imgfile)
    rawimage, image = importimage(imgpath)
    return image


def get_Random_Noise():
    noise = np.random.uniform(0, 1, [1, imagesize, imagesize, 3])

    return noise


def attack_untargeted(imgobj, alpha=0.2, beta=0.001, iterations=1000, max_query=5000):
    x0 = imgobj.img
    disttbl = []
    HistoryL2 = []
    HistoryLinf = []
    TimeHistory = [[0, 0]]
    t1 = millis()

    num_samples = 100
    best_theta, g_theta = None, float('inf')

    #print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    for i in range(num_samples):
        xi = get_Random_Noise()
        if imgobj.decision(xi):
            theta = xi - imgobj.img
            initial_lbd = np.linalg.norm(theta)
            theta = theta / np.linalg.norm(theta)
            lbd = fine_grained_binary_search(imgobj, theta, initial_lbd, g_theta)
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                #print("--------> Found distortion %.4f" % g_theta)
                t2 = millis()
                perturbed = x0 + g_theta * best_theta
                TimeHistory.append([imgobj.q, t2 - t1])
                twonorm, infnorm = norm(perturbed, imgobj.img)
                HistoryL2.append([imgobj.q, twonorm])
                HistoryLinf.append([imgobj.q, infnorm])
    #print("==========> Found best distortion %.4f using %d queries" % (twonorm, imgobj.q))

    timestart = time.time()
    g1 = 1.0
    theta, g2 = np.copy(best_theta), g_theta

    stopping = 0.01
    prev_obj = 100000
    t = tqdm(total=max_query)
    for i in range(iterations):
        t.n = imgobj.q
        t.update(n=0)
        #print('round:' + str(i))
        gradient = np.zeros(theta.shape)
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = np.random.randn(*theta.shape).astype('float')
            u = u / np.linalg.norm(u)
            ttt = theta + beta * u
            ttt = ttt / np.linalg.norm(ttt)
            g1 = fine_grained_binary_search_local(imgobj, ttt, initial_lbd=g2, tol=beta / 500)
            gradient += (g1 - g2) / beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0 / q * gradient

        if (i + 1) % 50 == 0:
            if g2 > prev_obj - stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2

        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta / np.linalg.norm(new_theta)
            new_g2 = fine_grained_binary_search_local(imgobj, new_theta, initial_lbd=min_g2, tol=beta / 500)
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta / np.linalg.norm(new_theta)
                new_g2 = fine_grained_binary_search_local(imgobj, new_theta, initial_lbd=min_g2, tol=beta / 500)
                if new_g2 < g2:
                    min_theta = new_theta
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = np.copy(theta), g2
        t2 = millis()
        perturbed = x0 + g_theta * best_theta
        TimeHistory.append([imgobj.q, t2 - t1])
        twonorm, infnorm = norm(perturbed, imgobj.img)
        HistoryL2.append([imgobj.q, twonorm])
        HistoryLinf.append([imgobj.q, infnorm])
        #print("==========> Found best distortion %.4f using %d queries" % (twonorm, imgobj.q))
        # print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            #print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

        if imgobj.q > max_query:
            break
    returning = x0 + g_theta * best_theta
    twonorm, infnorm = norm(imgobj.img, returning)
    return True,returning, [HistoryL2, HistoryLinf, TimeHistory],[twonorm,infnorm]


def fine_grained_binary_search_local(imgobj, theta, initial_lbd=1.0, tol=1e-5):
    lbd = initial_lbd
    x0 = imgobj.img
    if not imgobj.decision(x0 + lbd * theta):
        lbd_lo = lbd
        lbd_hi = lbd * 1.01
        while not imgobj.decision(x0 + lbd_hi * theta):
            lbd_hi = lbd_hi * 1.01
            if lbd_hi > 20:
                return float('inf')
    else:
        lbd_hi = lbd
        lbd_lo = lbd * 0.99
        while imgobj.decision(x0 + lbd_lo * theta):
            lbd_lo = lbd_lo * 0.99

    while (lbd_hi - lbd_lo) > tol:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if imgobj.decision(x0 + lbd_mid * theta):
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi


def fine_grained_binary_search(imgobj, theta, initial_lbd, current_best):
    x0 = imgobj.img
    if initial_lbd > current_best:
        if not imgobj.decision(x0 + current_best * theta):
            return float('inf')
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
        lbd_mid = (lbd_lo + lbd_hi) / 2.0
        if imgobj.decision(x0 + lbd_mid * theta):
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi
###HopSkipJumpAttak
from Util import get_imagenet_label,pretrained_model
def random_noise_hsja(imgobj):
    tries = 0
    while tries < 1000:
        tries += 1
        noise = np.random.uniform(0, 1, [1, imagesize, imagesize, 3])
        if imgobj.decision(noise):
            break

    lo = 0.0
    hi = 1.0
    while hi - lo > 0.001:

        mid = (hi + lo) / 2.0
        blended = (1 - mid) * imgobj.img + mid * noise
        if imgobj.decision(blended):
            hi = mid
        else:
            lo = mid

    final = (1 - hi) * imgobj.img + hi * noise
    return final


def select_delta(dist, l, cur_iter, theta, d):
    if cur_iter == 1:
        delta = 0.1
    else:
        if l == 'l2':
            delta = np.sqrt(d) * theta * dist
        elif l == 'linf':
            delta = np.sqrt(d) * theta * dist
    return delta


def geometric_progression(x, update, dist, cur_iter,imgobj):
    epsilon = dist / np.sqrt(cur_iter)

    def phi(epsilon):
        new = x + epsilon * update
        check1 = get_imagenet_label(pretrained_model.predict(new, steps=1))
        check2 = get_imagenet_label(pretrained_model.predict(x, steps=1))
        imgobj.q+=1#record the query
        success = [check1[0][0] != check2[0][0]]
        return success

    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon


def approximate_gradient(sample, num_evals, delta, l, imgobj):
    clip_max = 1
    clip_min = 0

    noise_shape = [num_evals] + list(sample.shape)
    if l == 'l2':
        rv = np.random.randn(*noise_shape)
    elif l == 'linf':
        rv = np.random.uniform(low=-1, high=1, size=noise_shape)

    rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))
    perturbed = sample + delta * rv
    perturbed = clip_image(perturbed, 0, 1)
    rv = (perturbed - sample) / delta

    decisions = np.array([])
    for i in range(num_evals):
        check1 = get_imagenet_label(pretrained_model.predict(perturbed[i], steps=1))
        check2 = get_imagenet_label(pretrained_model.predict(sample, steps=1))
        imgobj.q += 1#record the query
        boolean = [check1[0][0] != check2[0][0]]
        decisions = np.append(decisions, boolean)

    decision_shape = [len(decisions)] + [1] * len(sample.shape)
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

    if np.mean(fval) == 1.0:  # label changes.
        gradf = np.mean(rv, axis=0)
    elif np.mean(fval) == -1.0:  # label not change.
        gradf = - np.mean(rv, axis=0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * rv, axis=0)

        # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return -gradf


def binary_search_hsja(perturbed, imgobj, theta, l='l2'):
    distances = []
    for p in perturbed:
        if l == 'l2':
            distances.append(norm(p, imgobj.img)[0])
        else:
            distances.append(norm(p, imgobj.img)[1])

    if l == 'linf':
        highs = distances
        thresholds = np.minimum(np.asarray(distances) * theta, theta)
    else:
        highs = np.ones(len(perturbed))
        thresholds = theta

    lows = np.zeros(len(perturbed))

    while np.max((highs - lows) / thresholds) > 1:

        mids = (highs + lows) / 2.0

        decisions = np.array([])

        for p in range(len(perturbed)):
            mid_image = project(imgobj.img, perturbed[p], mids[p], l)
            d = imgobj.decision(mid_image)
            decisions = np.append(decisions, [d])

        # Update highs and lows based on model decisions.

        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    outputs = [project(imgobj.img, perturbed[p], highs[p], l) for p in range(len(perturbed))]

    finaldists = []
    for p in perturbed:
        if l == 'l2':
            finaldists.append(norm(p, imgobj.img)[0])
        else:
            finaldists.append(norm(p, imgobj.img)[1])

    idx = np.argmin(finaldists)

    dist = distances[idx]
    out_image = outputs[idx]
    return out_image, dist


def hsja(imgobj,  # instance of class randomimg
         constraint='l2',
         max_query=5000,
         num_iterations=100,
         gamma=10,
         max_num_evals=1e4,
         init_num_evals=100,
         verbose=True
         ):
    HistoryL2 = []
    HistoryLinf = []
    TimeHistory = [[0, 0]]
    best_f = float('inf')
    t1 = millis()
    d = np.prod(imgobj.img.shape)

    if constraint == 'l2':
        theta = gamma / d ** (3 / 2)
    else:
        theta = gamma / d ** 2

    perturbed = random_noise_hsja(imgobj)

    perturbed, dist_post = binary_search_hsja([perturbed], imgobj, theta, constraint)

    if constraint == 'l2':

        dist = norm(perturbed, imgobj.img)[0]
    else:
        dist = norm(perturbed, imgobj.img)[1]

    t = tqdm(total=max_query)
    for j in np.arange(num_iterations):
        c_iter = j + 1
        t.n = imgobj.q
        t.update(n=0)
        # Choose delta.
        delta = select_delta(dist, constraint, c_iter, theta, d)
        # Choose number of evaluations.
        num_evals = int(init_num_evals * np.sqrt(c_iter))
        num_evals = int(min([num_evals, max_num_evals]))

        # approximate gradient.
        gradf = approximate_gradient(perturbed, num_evals,
                                     delta, constraint, imgobj)

        if constraint == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf
        # find step size.
        epsilon = geometric_progression(perturbed,
                                        update, dist, c_iter,imgobj)
        # Update the sample.
        perturbed = clip_image(perturbed + epsilon * update, 0, 1)
        # Binary search to return to the boundary.
        perturbed, dist_post = binary_search_hsja(perturbed, imgobj, theta, constraint)
        # compute new distance.
        if constraint == 'l2':
            dist = norm(perturbed, imgobj.img)[0]
        else:
            dist = norm(perturbed, imgobj.img)[1]
        if verbose:
            print('queries: {:d}, iterations {:d}, {:s} distance {:.4E}'.format(imgobj.q, c_iter, constraint, dist))
        t2 = millis()
        TimeHistory.append([imgobj.q, t2 - t1])
        if dist<best_f:
            best_f=dist
            twonorm, infnorm = norm(perturbed, imgobj.img)
            HistoryL2.append([imgobj.q, twonorm])
            HistoryLinf.append([imgobj.q, infnorm])
        if imgobj.q > max_query:
            break
    return True, perturbed, [HistoryL2, HistoryLinf, TimeHistory],[twonorm,infnorm]

#Score-Based BO-attack
class preddifference_Score:
    def __init__(self, image, maxnorm,noise, constraint='l2',mode='Score'):
        self.image = image
        self.maxnorm = maxnorm
        self.noise=noise
        self.constraint=constraint
        self.mode=mode
        pass
    def func(self, parameters):
        final = create_distorted_image(self.image.img, self.noise, self.maxnorm/255, parameters)
        self.adv = final
        rawpredict = pretrained_model.predict(final)
        self.image.q += 1
        #print(self.image.q)
        predictions = get_imagenet_label(rawpredict)
        origprob = rawpredict[0][self.image.labelindex]
        return origprob-predictions[1][2]#<=0 means success


def bayesian_attack_Score(image, max_query, init_query=5, noise='perlin', max_norm=16,constraint='linf'):
    if noise == 'perlin':
        bounds = [{'name': 'wavelength', 'type': 'continuous', 'domain': (10, 200), 'dimensionality': 1},
                  {'name': 'octave', 'type': 'discrete', 'domain': (1, 2, 3, 4), 'dimensionality': 1},
                  {'name': 'freq_sine', 'type': 'continuous', 'domain': (4, 32), 'dimensionality': 1}
                  ]
    elif noise == 'gabor':
        bounds = [{'name': 'kernels', 'type': 'discrete', 'domain': (1, 200), 'dimensionality': 1},
                  {'name': 'kernel size', 'type': 'discrete', 'domain': (1, 40), 'dimensionality': 1},
                  {'name': 'sigma', 'type': 'continuous', 'domain': (1, 8), 'dimensionality': 1},
                  {'name': 'orientation', 'type': 'continuous', 'domain': (0, 2 * np.pi), 'dimensionality': 1},
                  {'name': 'scale', 'type': 'continuous', 'domain': (1, 20), 'dimensionality': 1},
                  {'name': 'sides', 'type': 'discrete', 'domain': (1, 12), 'dimensionality': 1}
                  ]
    else:
        bounds = [{'name': 'wavelength', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 48}]
    feasible_space = gy.Design_space(space=bounds)
    initial_design = gy.experiment_design.initial_design('random', feasible_space, init_query)

    queries = 0


    optimized = preddifference_Score(image, max_norm, noise=noise,constraint=constraint)

    # Gaussian process and Bayesian optimization
    objective = gy.core.task.SingleObjective(optimized.func, num_cores=1)
    model = gy.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
    aquisition_opt = gy.optimization.AcquisitionOptimizer(feasible_space)
    acquisition = gy.acquisitions.AcquisitionLCB(model, feasible_space, optimizer=aquisition_opt)
    evaluator = gy.core.evaluators.Sequential(acquisition, batch_size=1)
    BOpt = gy.methods.ModularBayesianOptimization(model, feasible_space, objective, acquisition, evaluator,
                                                  initial_design)
    HistoryL2 = []
    HistoryLinf = []
    bestDistanceL2 = 2550
    bestDistanceLinf = 2550
    best_f = float('inf')
    TimeHistory = [[0, 0]]
    t1 = millis()
    success = False
    last = 0
    t = tqdm(total=max_query)
    while image.q < max_query:
        BOpt.run_optimization(max_iter=1)
        t.n=image.q
        t.update(n=0)
        if image.q == last:
            break
        else:
            last = image.q
        t2 = millis()
        TimeHistory.append([image.q, t2 - t1])
        if BOpt.fx_opt <= 0:
            twonorm, infnorm = norm(image.img, optimized.adv)
            if infnorm < bestDistanceLinf:
                bestDistanceLinf = infnorm
                HistoryLinf.append([image.q, bestDistanceLinf])
                if constraint=='linf':
                    success=True
                    final = optimized.adv
            if twonorm < bestDistanceL2:
                bestDistanceL2 = twonorm
                HistoryL2.append([image.q, bestDistanceL2])
                if constraint=='l2':
                    success=True
                    final = optimized.adv
    t.close()
    HistoryL2.append([image.q, bestDistanceL2])
    HistoryLinf.append([image.q, bestDistanceLinf])
    if success:
        twonorm, infnorm = norm(image.img, final)
    else:
        final=None
        twonorm=bestDistanceL2
        infnorm=bestDistanceLinf
    return success,final, [HistoryL2, HistoryLinf, TimeHistory],[twonorm,infnorm]