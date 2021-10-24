import time
import numpy as np
import torch
import torch.nn.functional as F
from Util import millis
from tqdm.notebook import tqdm

def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.

    return np.minimum(np.maximum(clip_min, image), clip_max)
class RayS(object):
    def __init__(self, randomimg, order=np.inf, epsilon=0.3, early_stopping=False):
        self.RandomImg = randomimg
        self.order = order
        self.epsilon = epsilon
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}
        self.early_stopping = early_stopping

    def get_xadv(self, x, v, d, lb=0., rb=1.):
        out = x + d * v
        out = clip_image(out, 0, 1)
        return out

    def attack_hard_label(self,target=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            (x, y): original image
        """
        x = torch.from_numpy(self.RandomImg.img.numpy())
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)
        self.d_t = np.inf
        self.sgn_t = torch.sign(torch.ones(shape))
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        dist = torch.tensor(np.inf)
        block_level = 0
        block_ind = 0
        TimeHistory = [[0, 0]]
        t1 = millis()
        t = tqdm(total=query_limit)
        while self.RandomImg.q < query_limit:
            t.n = self.RandomImg.q
            t.update(n=0)

            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, attempt)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm(self.x_final - x, self.order)
            if self.early_stopping and (dist <= self.epsilon):
                break
            t2=millis()
            TimeHistory.append([self.RandomImg.q, t2 - t1])

        return TimeHistory,self.x_final.numpy()

    def search_succ(self, x):
        return self.RandomImg.decision(x.numpy())

    def lin_search(self, x, sgn):
        d_end = np.inf
        for d in range(1, self.lin_search_rad + 1):
            if self.search_succ(self.get_xadv(x, sgn, d)):
                d_end = d
                break
        return d_end

    def binary_search(self, x, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t)):
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.lin_search(x, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(x, sgn_unit, d_mid)):
                d_end = d_mid
            else:
                d_start = d_mid
        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False
