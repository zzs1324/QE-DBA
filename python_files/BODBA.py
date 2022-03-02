import pickle
import numpy as np
import colorsys
import tensorflow as tf
from python_files.Util import imagesize, norm
import GPyOpt as gy
import noise as ns
import cv2
from tqdm.notebook import tqdm
from python_files.Util import millis, mod



def SAVE(fp, input):
    with open(fp, "wb+") as fp:
        pickle.dump(input, fp)
    return


def LOAD(fp):
    with open(fp, "rb+") as fp:
        output = pickle.load(fp)
    return output


def rgb2gray(rgb):
    gray = np.mean(rgb, -1)

    return gray


def binary_search(perturbed_image, imgobj, theta, l='l2'):
    low = 0
    while imgobj.decision(perturbed_image) == 0:
        lowimage = perturbed_image
        perturbed_image = (perturbed_image - imgobj.img) * 2 + imgobj.img
        perturbed_image = np.clip(perturbed_image, 0, 1)
        if np.array_equal(lowimage, perturbed_image):
            print('inf happened')
            if l == 'l2':
                return perturbed_image, imagesize * imagesize
            else:
                return perturbed_image, 1.0
        #        display_images(perturbed_image)
        low = 0
    high = 1


    # print(str(high)+','+str(low))
    while (high - low) / theta > 1:
        #        print(imgobj.q,end=";")
        mid = (high + low) / 2.0
        mid_image = project(imgobj.img, perturbed_image, mid)
        d = imgobj.decision(mid_image)
        if d == 1:
            high = mid
        else:
            low = mid
        # print(str(high)+','+str(low))

    output = project(imgobj.img, perturbed_image, high)

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





def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)


def project(original_image, perturbed_images, alphas):

    return (1 - alphas) * original_image + alphas * perturbed_images


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
    if mod == 'MNIST':
        blank = np.expand_dims(rgb2gray(blank), -1)
        # print(blank.shape)
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
    if mod == 'MNIST':
        return np.expand_dims(normn, -1)
    else:
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


from python_files.Upsample import upsample_projection


def create_distorted_image(image, typ, epsilon, parameters):
    parameters = parameters[0]
    if typ == 'perlin':
        pert = perlin_noise(parameters[0], parameters[1], parameters[2])
    elif typ == 'gabor':
        pert = gabor_noise_random(int(parameters[0]), int(parameters[1]), parameters[2], parameters[3], parameters[4],
                                  sides=int(parameters[5]))
    elif typ == 'BICU' or typ == 'CLUSTER' or typ == 'NN' or typ == 'BILI':
        parameters = np.expand_dims(parameters, axis=0)
        pert = upsample_projection(typ, parameters, 16, imagesize * imagesize, nchannel=3)
        #pert = np.repeat(pert,3,axis=0)
        pert = pert.reshape(1, 3, imagesize, imagesize)
        pert = pert.transpose(0, 2, 3, 1)
        pert = (pert - np.min(pert)) / np.ptp(pert)

    pert = (pert - .5) * 2
    #print(str(np.max(pert))+','+str(np.min(pert)))
    dist_img = image + epsilon * pert
    return dist_img


class preddifference:
    def __init__(self, image, maxnorm, noise, constraint='l2'):
        self.image = image
        self.maxnorm = maxnorm
        self.noise = noise
        self.best = imagesize * imagesize
        self.theta = 0.01
        self.adv = None
        self.constraint = constraint
        pass

    def func(self, parameters):
        # print(parameters)
        final = create_distorted_image(self.image.img, self.noise, self.maxnorm / 255, parameters)
        # print(final.shape)
        out, dist = binary_search(final, self.image, self.theta, l=self.constraint)
        # print(out.shape)
        if dist < self.best:
            self.best = dist
            self.adv = out
        return dist  # np.log(dist*(255/self.maxnorm))


def bayesian_attack(image, max_query, init_query=5, noise='perlin', max_norm=16, constraint='l2'):
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

    optimized = preddifference(image, max_norm, noise=noise, constraint=constraint)

    # Gaussian process and Bayesian optimization
    objective = gy.core.task.SingleObjective(optimized.func, num_cores=1)
    model = gy.models.GPModel(exact_feval=False, optimize_restarts=5, verbose=False)
    aquisition_opt = gy.optimization.AcquisitionOptimizer(feasible_space)
    acquisition = gy.acquisitions.AcquisitionEI(model, feasible_space, optimizer=aquisition_opt)
    evaluator = gy.core.evaluators.Sequential(acquisition, batch_size=1)
    BOpt = gy.methods.ModularBayesianOptimization(model, feasible_space, objective, acquisition, evaluator,
                                                  initial_design)

    TimeHistory = [[0, 0]]
    t1 = millis()
    last = 0
    t = tqdm(total=max_query)
    quitcount = 0
    while image.q < max_query:
        BOpt.run_optimization(max_iter=1)
        t.n = image.q
        t.update(n=0)
        if image.q == last:
            quitcount = quitcount + 1
            if quitcount < 5:
                print('===============')
                continue
            else:
                break
        else:
            last = image.q
        t2 = millis()
        TimeHistory.append([image.q, t2 - t1])
    return TimeHistory, optimized.adv
