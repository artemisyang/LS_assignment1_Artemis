from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts
    
def sim_health_index():

    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
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

    # Evenly distribute number of simulation runs across processes
    N = int(S/size)

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

  # Gather all simulation arrays to buffer on rank 0
    z_mat_all = None
    if rank == 0:
       z_mat_all = np.empty([T, int(N*size)], dtype = 'float')
     #  z_mat_all = np.zeros((T, int(N*size)))
     #  z_mat_all = np.empty([S, T], dtype='float')
    comm.Gather(sendbuf=z_mat, recvbuf=z_mat_all, root=0)
    #   comm.gather(z_mat, root=0)
     
  # Print simulation results on rank 0
    if rank == 0:
     # Calculate time elapsed
       time_elapsed = time.time() - t0
     # Print(time_elapsed)
       print("Simulated lifetimes in: %f seconds on %d MPI processes"
                % (time_elapsed, size))     
   
    return

def main():
    sim_health_index()

if __name__ == '__main__':
    main()
   
