from math import lgamma
import numpy as np
from scipy.special import digamma

def tune_hyper(data, hyper, n_iterations = 1000):
  # data is assumed to be n_sample * n_features
  # hyper is assumed to list of n_features
  n_samples, n_features = data.shape
  
  # setup a list for storing the previous iterations hypers
  current_hyper = np.copy(hyper)
  # sum all the rows in data
  sample_sums = data.sum(axis=1)

  # preform the fixed point iteration
  e = 0
  for i in range(n_iterations):
    prev_hyper = np.copy(current_hyper)
    sum_prev_hyper = sum(prev_hyper)


    # calculate sum_g for every f 
    sum_g = np.sum(digamma(data + np.tile(prev_hyper, (n_samples, 1))), axis=0)
    # calcualte sum_h for every f
    sum_h = sum(digamma(sample_sums + sum_prev_hyper))
    # calcualte new hyper-value for each feature
    current_hyper = prev_hyper * (sum_g - n_samples * digamma(prev_hyper))\
                                / (sum_h - n_samples * digamma(sum_prev_hyper))
    
    # check every dimesion of hyper to see if it has converged
    done = False
    e = 0
    for p, c in zip(prev_hyper, current_hyper):
      e += abs(p - c)
    if e <= 0.000000001:
      done = True
      # TODO add early stopping
    #if abs(e - le) < 0.000001:
    #  done = True

    if done:
      print("Completed polya fit at iteration " + str(i))
      break

  return current_hyper
