import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.clrandom as clrand
import pyopencl.tools as cltools
from pyopencl.elementwise import ElementwiseKernel
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts

def sim_health_index(r):

  # Set up context and command queue
  ctx = cl.create_some_context()
  queue = cl.CommandQueue(ctx)
    
  # Start time:
  t0 = time.time()
   
  # Set model parameters
  S = 1000 # Set the number of lives to simulate
  T = int(4160) # Set the number of periods for each simulation
  rho = r
  mu = 3.0
  np.random.seed(25)
  z_mat = np.zeros((T, S), dtype=np.float32)
  
  # Generate array of random shocks
  init_np = np.zeros(S).astype(np.float32) + mu
  epsm_np = sts.norm.rvs(loc=0,scale=1.0,size=T*S).astype(np.float32)

  init_g = cl.array.to_device(queue, init_np)
  epsm_g = cl.array.to_device(queue, epsm_np)
  
  # GPU: Define Segmented Elementwise Kernel
  prefix_sum = ElementwiseKernel(ctx,
                                 "float *a_g, float *b_g, float *res_g, float rho, float mu",
                                 "res_g[i] = rho * a_g[i]+(1-rho)*mu + b_g[i]")
  
  # Allocate space for result of kernel on device
  dev_result = cl_array.empty_like(epsm_g)

  # Enqueue and Run Elementwise Kernel
  prefix_sum(init_g, epsm_g[:S], dev_result[:S], rho, mu)
  [prefix_sum(dev_result[S*(i-1):S*i], epsm_g[S*i:S*(i+1)],
              dev_result[S*i:S*(i+1)], rho, mu) for i in range(1,T)]

  # Get results back on CPU
  z_all = dev_result.get().reshape(T,S)

  # Create an array to store the index for first negative z_t
  neg_index = np.full(S,fill_value = T+1)

  # Print simulation results
  for s in range(S):
    for t in range(T):
      if z_all[t,s] < 0:
        if neg_index[s] == T+1:
           neg_index[s] = t+1
  mean = np.mean(neg_index)
  return mean

def main():
    # Start time
    t0 = time.time()
    
    # Run the health simulation program
    rho_array = np.linspace(-0.95,0.95,200)
    avg =[]
    for r in rho_array:
        avg.append(sim_health_index(r))

    # Get optimal rho value
    ind = avg.index(max(avg))
    opt = rho_array[ind]
    max_avg = max(avg)
    
    # End time
    time_elapsed = time.time() - t0

    # Print simulation results
    print("Simulated in %f seconds, Optimal rho value = %f, Average periods = %f"
                % (time_elapsed, opt, max_avg)) 
  
if __name__ == '__main__':
    main()
