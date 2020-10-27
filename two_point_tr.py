import numpy as np
import os
import pandas as pd
import pickle as pk
import time 

folder = './Mat_G_0_75/'
files = [folder+elem for elem in os.listdir(folder) if 'exp' in elem]
f_count = len(files)

start = time.time()

ensX = pd.DataFrame()
ensY = pd.DataFrame()

for file in files:
    inner_start = time.time()
    f_count -= 1
    mX = np.load(file)
    mY = mX.T
    DIM_row = mX.shape[0]
    DIM_col = mX.shape[1]
    freq_mX = np.zeros(DIM_col-1)
    freq_mY = np.zeros(DIM_col-1)
    div = np.arange(DIM_col-1,0,-1)
    for i in range(DIM_row):
        ccX = np.correlate(mX[i],mX[i],'full')[DIM_col:] / div
        ccY = np.correlate(mY[i],mY[i],'full')[DIM_col:] / div
        freq_mX += ccX
        freq_mY += ccY
    
    freq_mX /= DIM_row
    freq_mY /= DIM_row

    ensX[file] = freq_mX
    ensY[file] = freq_mY
    inner_end = time.time()
    print('processing:',file,'---',f_count,'files left --- iter time:',inner_end - inner_start,'sec')

ensX.index = np.arange(1,DIM_col)
ensY.index = np.arange(1,DIM_col)

end = time.time()
print('processing time:', end - start,'sec')

with open('./s2_all_X_0_75.pickle','wb') as f1:
    pk.dump(ensX,f1)

with open('./s2_mean_X_0_75.pickle','wb') as f2:
    pk.dump(ensX.mean(axis=1),f2)

with open('./s2_all_Y_0_75.pickle','wb') as f3:
    pk.dump(ensY,f3)

with open('./s2_mean_Y_0_75.pickle','wb') as f4:
    pk.dump(ensY.mean(axis=1),f4)
