import numpy as np
import os
import pandas as pd
import pickle as pk
import time 

folder = './Mat_G/'
matrixes = {}
start_load = time.time()
for filename in os.listdir(folder):
    if filename.endswith('.npy'):
        print('loading:',filename)
        matrixes[filename] = np.load(folder+filename)
end_load = time.time()
print('load time is:', end_load - start_load,'sec')

start_calc = time.time()
ens = pd.DataFrame()
for key in matrixes:
    print('processing:',key)
    m = matrixes[key]
    DIM_row = m.shape[0]
    DIM_col = m.shape[1]
    freq_m = np.zeros(DIM_col-1)
    div = np.arange(DIM_col-1,0,-1)
    for i in range(DIM_row):
        cc = np.correlate(m[i],m[i],'full')[DIM_col:] / div
        freq_m += cc
    freq_m /= DIM_row
    ens[key] = freq_m
ens.index = np.arange(1,DIM_col)
end_calc = time.time()
print('processing time:', end_calc - start_calc,'sec')

with open('./s2_all.pickle','wb') as f1:
    pk.dump(ens,f1)

with open('./s2_mean.pickle','wb') as f2:
    pk.dump(ens.mean(axis=1),f2)