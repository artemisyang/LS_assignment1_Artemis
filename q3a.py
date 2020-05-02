from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts
    
def sim_health_index():

 # Start time
    t0 = time.time()
    
    # Run the health simulation program
    rho_array = np.linspace(-0.95,0.95,200)
    avg =[]
    
    for r in rho_array:
    
        # Get rank of process and overall size of communicator:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
   
        # Set model parameters
        rho = r
        mu = 3.0
        sigma = 1.0
        z_0 = mu

        # Set simulation parameters
        S = 10 # Set the number of lives to simulate
        T = int(410) # Set the number of periods for each simulation

        # Evenly distribute number of simulation runs across processes
        N = int(S/size)

        # Draw all idiosyncratic random shocks and create empty containers
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
     
        # Return average number of periods on rank 0
        if rank == 0:
           mean = np.mean(neg_all)
           avg.append(mean)
           
    if rank == 0:
        
       # Get optimal rho value
       ind = avg.index(max(avg))
       opt = rho_array[ind]
       max_avg = max(avg)
    
       # End time
       time_elapsed = time.time() - t0

       # Print simulation results
       print("Simulated in %f seconds, Optimal rho value = %f, Average periods = %f"
                % (time_elapsed, opt, max_avg)) 
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy.stats as sts
    
def sim_health_index():

 # Start time
    t0 = time.time()
    
    # Run the health simulation program
    rho_array = np.linspace(-0.95,0.95,10)
    avg =[]
    
    for r in rho_array:
    
        # Get rank of process and overall size of communicator:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
   
        # Set model parameters
        rho = r
        mu = 3.0
        sigma = 1.0
        z_0 = mu

        # Set simulation parameters
        S = 10 # Set the number of lives to simulate
        T = int(40) # Set the number of periods for each simulation

        # Evenly distribute number of simulation runs across processes
        N = int(S/size)

        # Draw all idiosyncratic random shocks and create empty containers
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
     
        # Return average number of periods on rank 0
        if rank == 0:
           mean = np.mean(neg_all)
           avg.append(mean)
           
    if rank == 0:
        
       # Get optimal rho value
       ind = avg.index(max(avg))
       opt = rho_array[ind]
       max_avg = max(avg)
    
       # End time
       time_elapsed = time.time() - t0

       # Print simulation results
       print("Simulated in %f seconds, Optimal rho value = %f, Average periods = %f"
                % (time_elapsed, opt, max_avg)) 
       print(rho_array)
       print(avg)

def main():
   sim_health_index()
          
if __name__ == '__main__':
    main() 

def main():
   sim_health_index()
          
if __name__ == '__main__':
    main()
