import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.scan import GenericScanKernel
import matplotlib.pyplot as plt
import time
import scipy.stats as sts

def sim_health_index(n_runs):

  # Set up context and command queue
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx)
    
  # Start time:
  t0 = time.time()
   
  # Set model parameters
  rho = 0.5
  mu = 3.0
  sigma = 1.0
  z_0 = mu

  # Set simulation parameters
  S = 1000 # Set the number of lives to simulate
  T = int(4160) # Set the number of periods for each simulation

  # Draw all idiosyncratic random shocks and create empty containers
  np.random.seed(25)
  eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
  z_mat = np.zeros((T, N))
  z_mat[0, :] = z_0

  for s_ind in range(N):
    z_tm1 = z_0
    for t_ind in range(T):
      e_t = eps_mat[t_ind, s_ind]
      z_t = rho * z_tm1 + (1 - rho) * mu + e_t
      z_mat[t_ind, s_ind] = z_t
      z_tm1 = z_t

  # Print simulation results
    print("Simulated %d Random Walks in: %f seconds"
                % (n_runs, time_elapsed)) 

def main():
    sim_health_index(n_runs = 1000)

if __name__ == '__main__':
    main()