from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

t0 = time.time()

#  rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters
S = 1000 # Set the number of lives to simulate
T = int(4160) # Set the number of periods for each simulation

# Evenly distribute number of simulation runs across processes
N = int(S/size)

def parallel_function_caller(x,stopp):
    stopp[0]=comm.bcast(stopp[0], root=0)
    summ=0
    if stopp[0]==0:
        #your function here in parallel
        x=comm.bcast(x, root=0)
        #Create an array to store the index for first negative z_t
        neg_index = np.full(N,fill_value = T+1)
        # Draw all idiosyncratic random shocks and create empty containers
        # np.random.seed(25)
        eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
        z_mat = np.zeros((T, N))
        for s_ind in range(N):
            z_tm1 = z_0
            for t_ind in range(T):
                e_t = eps_mat[t_ind, s_ind]
                z_t = x[0] * z_tm1 + (1 - x[0]) * mu + e_t
                z_mat[t_ind, s_ind] = z_t
                z_tm1 = z_t
                if z_tm1<0:
                  if neg_index[s_ind] == T+1:
                     neg_index[s_ind] = t_ind+1 
        neg_all = comm.gather(neg_index, root = 0)   
        #print(np.mean(neg_all)) 
        summl=np.max(np.mean(neg_all))
        summ=comm.reduce(summl,op=MPI.SUM, root=0)
        if rank==0:
          print("rho value is "+str(x)+", average periods is "+str(summ))
    return -summ

if rank == 0 :
   stop=[0]
   x = np.zeros(1)
   x[0]=0.1
   xs = minimize(parallel_function_caller,x0 = x, args=(stop,), 
                 method = 'L-BFGS-B', bounds = ((-0.95,0.95),), 
                 options = {'eps' : 0.2}) #method = 'COBYLA'
   print("the argmin is "+str(xs))
   stop=[1]
   parallel_function_caller(x,stop)

else :
   stop=[0]
   x=np.zeros(1)
   while stop[0]==0:
      parallel_function_caller(x,stop)

time_elapsed = time.time() - t0
print("Simulated in %f seconds" % (time_elapsed)) 
