# pylint: disable=E1101, E0401, E1102
import time
import random
import argparse
import pickle
import numpy as np
import torch
import sys

from Bayes_util import proj, latent_proj, transform

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, PosteriorMean
from botorch.acquisition import ProbabilityOfImprovement, UpperConfidenceBound
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import joint_optimize, gen_batch_initial_conditions
from botorch.gen import gen_candidates_torch, get_best_candidates

from tqdm.notebook import tqdm
from Util import millis, mod


parser = argparse.ArgumentParser()
# network architecture (resnet50, vgg16_bn, or inception_v3)
parser.add_argument('--arch', type=str, default=mod)#Changed to consists with our Util.py
# BayesOpt acquisition function
parser.add_argument('--acqf', type=str, default='EI')
# hyperparam for UCB acquisition function
parser.add_argument('--beta', type=float, default=1.0)
# number of channels in image
parser.add_argument('--channel', type=int, default=3)
parser.add_argument('--dim', type=int, default=12)  # dimension of attack
# dataset to attack
parser.add_argument('--dset', type=str, default='CIFAR')#Changed to consists with our code
# if True, project to boundary of epsilon ball (instead of just projecting inside)
parser.add_argument('--discrete', default=False, action='store_true')
# bound on perturbation norm
parser.add_argument('--eps', type=float, default=16/255.0)
# hard-label vs soft-label attack
parser.add_argument('--hard_label', default=True, action='store_true')
# number of BayesOpt iterations to perform
parser.add_argument('--iter', type=int, default=495)
# number of samples taken to form the GP prior
parser.add_argument('--initial_samples', type=int, default=5)
parser.add_argument('--inf_norm', default=True,
                    action='store_true')  # perform L_inf norm attack
# number of images to attack
parser.add_argument('--num_attacks', type=int, default=1)
# hyperparam for acquisition function
parser.add_argument('--num_restarts', type=int, default=1)
# backend for acquisition function optimization (torch or scipy)
parser.add_argument('--optimize_acq', type=str, default='torch')
# number of candidates to receive from acquisition function
parser.add_argument('--q', type=int, default=1)
# index of first image to attack
parser.add_argument('--start', type=int, default=0)
# save dictionary of results at end
parser.add_argument('--save', default=False, action='store_true')
parser.add_argument('--standardize', default=False,
                    action='store_true')  # normalize objective values
# normalize objective values at every BayesOpt iteration
parser.add_argument('--standardize_every_iter',
                    default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1)  # random seed
# if True, use sine FFT basis vectors
parser.add_argument('--sin', default=True, action='store_true')
# if True, use cosine FFT basis vectors
parser.add_argument('--cos', default=True, action='store_true')
args = parser.parse_args([])
get_best_candidates

if args.sin and args.cos:
    latent_dim = args.dim * args.dim * args.channel * 2
else:
    latent_dim = args.dim * args.dim * args.channel

device = torch.device('cpu')
bounds = torch.tensor([[-2.0] * latent_dim, [2.0] * latent_dim]).float()


def obj_func(x, randomimg):
    # evaluate objective function
    # if hard label: -1 if image is correctly classified, 0 otherwise
    # (done this way because BayesOpt assumes we want to maximize)
    # if soft label, correct logit - highest logit other than correct logit
    # in both cases, successful adversarial perturbation iff objective function >= 0
    #print(x.shape)
    x = transform(x, args.dset, args.arch, args.cos, args.sin)
    x = proj(x, args.eps, args.inf_norm, args.discrete)
    x = x.numpy()
    results=[]
    for i in range(x.shape[0]):
        temp = x[i]
        temp = np.moveaxis(temp, 0, -1)
        if randomimg.decision(np.expand_dims(temp,0)):
            res=0
        else:
            res=-1  
        results.append(res)
    print(results)
    return torch.FloatTensor(results)
        


def initialize_model(randomimg, n=5):
    # initialize botorch GP model

    # generate prior xs and ys for GP
    train_x = 2 * torch.rand(n, latent_dim).float() - 1
    if not args.inf_norm:
        train_x = latent_proj(train_x, args.eps)
    train_obj = obj_func(train_x, randomimg)
    mean, std = train_obj.mean(), train_obj.std()
    if args.standardize:
        train_obj = (train_obj - train_obj.mean()) / train_obj.std()
    best_observed_value = train_obj.max().item()
    # define models for objective and constraint
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj[:, None])
    model = model.to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = mll.to(train_x)
    return train_x, train_obj, mll, model, best_observed_value, mean, std


def optimize_acqf_and_get_observation(acq_func, randomimg):
    # Optimizes the acquisition function, returns new candidate new_x
    # and its objective function value new_obj

    # optimize
    if args.optimize_acq == 'scipy':
        batch_candidates, batch_acq_values = joint_optimize(
            acq_function=acq_func,
            bounds=bounds,
            q=args.q,
            num_restarts=args.num_restarts,
            raw_samples=200,
        )
        candidates = get_best_candidates(batch_candidates, batch_acq_values)
    else:
        Xinit = gen_batch_initial_conditions(
            acq_func,
            bounds,
            q=args.q,
            num_restarts=args.num_restarts,
            raw_samples=500
        )
        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=Xinit,
            acquisition_function=acq_func,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            verbose=False
        )
        candidates = get_best_candidates(batch_candidates, batch_acq_values)

    # observe new values
    new_x = candidates.detach()
    if not args.inf_norm:
        new_x = latent_proj(new_x, args.eps)
    new_obj = obj_func(new_x, randomimg)
    return new_x, new_obj

""" def optimize_acqf_and_get_observation(acq_func, randomimg):
    # Optimizes the acquisition function, returns new candidate new_x
    # and its objective function value new_obj

    # optimize
    if args.optimize_acq == 'scipy':
        batch_candidates,batch_acq_values = joint_optimize(
            acq_function=acq_func,
            bounds=bounds,
            q=args.q,
            num_restarts=args.num_restarts,
            raw_samples=200,
        )
        candidates = get_best_candidates(batch_candidates, batch_acq_values)
    else:
        Xinit = gen_batch_initial_conditions(
            acq_func,
            bounds,
            q=args.q,
            num_restarts=args.num_restarts,
            raw_samples=500
        )
        batch_candidates, batch_acq_values = gen_candidates_scipy(
            initial_conditions=Xinit,
            acquisition_function=acq_func,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            verbose=False
        )
        candidates = get_best_candidates(batch_candidates, batch_acq_values)

    # observe new values
    new_x = candidates
    print(new_x)
    if not args.inf_norm:
        new_x = latent_proj(new_x, args.eps)
    print(new_x)
    new_obj = obj_func(new_x, randomimg)
    return new_x, new_obj
 """

def bayes_opt(randomimg,constraint,max_query):
    """
    Main Bayesian optimization loop. Begins by initializing model, then for each
    iteration, it fits the GP to the data, gets a new point with the acquisition
    function, adds it to the dataset, and exits if it's a successful attack
    """
    x0=randomimg.img[0]
    x0 = np.moveaxis(x0, -1, 0)
    args.iter= max_query
    if constraint=='linf':
        args.inf_norm=True
    elif constraint=='l2':
        args.inf_norm=False
    print(args)
    best_observed = []
    query_count, success = 0, 0
    
    # call helper function to initialize model
    train_x, train_obj, mll, model, best_value, mean, std = initialize_model(
        randomimg, n=args.initial_samples)
    #print('initialization finished')
    if args.standardize_every_iter:
        train_obj = (train_obj - train_obj.mean()) / train_obj.std()
    best_observed.append(best_value)
    query_count += args.initial_samples

    # run args.iter rounds of BayesOpt after the initial random batch
    for count in range(args.iter):
        #print(count)
        # fit the model
        fit_gpytorch_model(mll)

        # define the qNEI acquisition module using a QMC sampler
        if args.q != 1:
            qmc_sampler = SobolQMCNormalSampler(num_samples=2000,
                                                seed=seed)
            qEI = qExpectedImprovement(model=model, sampler=qmc_sampler,
                                       best_f=best_value)
        else:
            if args.acqf == 'EI':
                qEI = ExpectedImprovement(model=model, best_f=best_value)
            elif args.acqf == 'PM':
                qEI = PosteriorMean(model)
            elif args.acqf == 'POI':
                qEI = ProbabilityOfImprovement(model, best_f=best_value)
            elif args.acqf == 'UCB':
                qEI = UpperConfidenceBound(model, beta=args.beta)
        #print('optimize and get new observation')
        # optimize and get new observation
        
        new_x, new_obj = optimize_acqf_and_get_observation(qEI, randomimg)

        if args.standardize:
            new_obj = (new_obj - mean) / std

        # update training points
        train_x = torch.cat((train_x, new_x))
        train_obj = torch.cat((train_obj, new_obj))
        if args.standardize_every_iter:
            train_obj = (train_obj - train_obj.mean()) / train_obj.std()

        # update progressargs
        best_value, best_index = train_obj.max(0)
        best_observed.append(best_value.item())
        best_candidate = train_x[best_index]

        # reinitialize the model so it is ready for fitting on next iteration
        torch.cuda.empty_cache()
        model.set_train_data(train_x, train_obj, strict=False)

        # get objective value of best candidate; if we found an adversary, exit
        best_candidate = best_candidate.view(1, -1)
        best_candidate = transform(
            best_candidate, args.dset, args.arch, args.cos, args.sin)
        best_candidate = proj(best_candidate, args.eps,
                              args.inf_norm, args.discrete)

        temp = best_candidate.numpy()[0] + x0
        temp = np.moveaxis(temp, 0, -1)
        if randomimg.decision(np.expand_dims(temp,0)):
            success = 1
            if args.inf_norm:
                print('Norm:', best_candidate.abs().max().item())
            else:
                print('Norm:', best_candidate.norm().item())
            return query_count,success, temp
        query_count += args.q
    # not successful (ran out of query budget)
    return query_count,success, temp
