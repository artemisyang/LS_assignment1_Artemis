from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts
    
def sim_health_index(r):

    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Start time:
    t0 = time.time()
   
    # Set model parameters
    rho = r
    mu = 3.0
    sigma = 1.0
    z_0 = mu

    # Set simulation parameters
    S = 1000 # Set the number of lives to simulate
    T = int(4160) # Set the number of periods for each simulation

    # Evenly distribute number of simulation runs across processes
    N = int(S/size)

    # Draw all idiosyncratic random shocks and create empty containers
    np.random.seed(25)
    eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, N))
    z_mat = np.zeros((T, N))
    z_mat[0, :] = z_0
    
    # Create an array to store the index for first negative z_t
    neg_index = np.full(N,fill_value = T+1)

    # Create the health index matrix
    for s_ind in range(N):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
            if z_tm1<0:
              if neg_index[s_ind] == T+1:
                 neg_index[s_ind] = t_ind+1
                 
  # Gather all simulation arrays to buffer on rank 0
    neg_all = comm.gather(neg_index, root = 0)
     
  # Return simulation results on rank 0
    if rank == 0:
       mean = np.mean(neg_all)
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
    
    # End time
    time_elapsed = time.time() - t0

    # Print simulation results
    print("Simulated in %f seconds, found optimal rho value = %f"
                % (time_elapsed, opt)) 
  
if __name__ == '__main__':
    main()
