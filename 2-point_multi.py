import numpy as np
import os
import time
import pickle as pk
from multiprocessing import Pool


def chunks(l, n):
    """Yield n number of striped chunks from list"""
    for i in range(0, n):
        yield l[i::n]

def S2_calc(files):
    """ 
    Calculate the 2-point correlation for a matrix representing the values
    of an indicator function of solid/fluid fraction in a porous media. 
    Inputs: the data files and the discretization pace. 
    Output: the S2 vector for each distance over all the rows of the matrix. 
    """

    # Discretization length dx = 4 micro m
    dx = 1 # the pace of the discretization

    # array which store the mean value of S_2 for each distance 'r'
    S2_array = []

    for file in files:

        print('processing',file)
        # load matrix
        m = np.load(file)

        freq_m = [] # Initialize freq array

        # initialize the distances between points
        r = [dx * elem for elem in range(1,m.shape[0])]

        # Rank of the matrix
        DIM = m.shape[0]

        # Cycle over all row of matrix
        for i in range(DIM):
            v = m[i]

            # calculate freq of ones over the range of distances 'r'
            freq_m.append([ sum(np.multiply(v[d:],v[:DIM-d]))/(DIM-d) for d in r])

        # average value of S_2 for each distance over all rows of matrix    
        S2 = np.mean(freq_m, axis=0)

        # store S_2 correlation of each matrix
        S2_array.append(S2)

    return S2_array

# number of cpu
nn = 4

# The geometry data are in the same location
# as the script

dir_list = os.listdir()
files = [elem for elem in dir_list if 'exp' in elem]

# Divide the files list in chunks
cc = list(chunks(files, nn))

# Start a pool of processes
p = Pool(nn)

# output of each subprocess is saved in the list 'output'
start_time = time.time()

output = p.map(S2_calc, cc)
p.close()
print("--- %s seconds ---" % (time.time() - start_time))


# merge the outputs
S2_array = np.array([j for sub in output for j in sub])

# calculate the ensemble average over all matrixes    
S2_ensamble = np.mean(S2_array, axis=0)

# Save the S_2 data for each matrix in file
with open('S2_array.pickle','wb') as S2_arr_file:
    pk.dump(S2_array,S2_arr_file)

# Save the ensable average for each distance in file
with open('S2_ensamble.pickle','wb') as S2_ens_file:
    pk.dump(S2_ensamble,S2_ens_file)
