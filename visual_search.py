"""
    Written by Shima Rashidi
    Wed 21 Oct 2020 19:51:08 AEDT
"""
import cv2
from PIL import Image
from math import *
from random import *

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import matplotlib.gridspec as gridspec

import torch
import torchvision
from torchvision import models

import pandas as pd
import os
import glob
from copy import deepcopy
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import minimize
from scipy import stats

from functions import get_feature_vecs
from functions import makepatches
from functions import paste_person
from functions import my_log
from functions import plot_classifier_output_hist
from functions import exp_func



def dist(x,y):
    # calculates the euiclidian distance of two vectors
    # @ param x: np array
    # @ param y: np array
    return sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def get_loc_tri_arr():
    # gets the 84 locations on a 7 deg circle on equi-distance triangels
    # outputs the 84 locations in an array
    listx = []
    listy = []
    a = 66 #the distance of each two point (666/2)/5 (5 points on each radius)
    loc_list = []
    for i in range(-6, 7):
        for j in range(-5, 6):
            x = (j) * a + (i % 2) * a / 2
            y = i * (sqrt(a ** 2 - (a / 2) ** 2))
            listx.append(x)
            listy.append(y)

    r = 313 #the center of the image(333) - half of tagret size (20)
    counter = 1
    listy1 = []
    listx1 = []
    for i in range(0, np.shape(listx)[0]):
        if listx[i] ** 2 + listy[i] ** 2 < (counter * r) ** 2:
            listx1.append(int(listx[i]) + 313)
            listy1.append(int(listy[i]) + 313)
            if int(listx[i])+ 313==313 and int(listy[i])+ 313==313:
                print('hi')
            else:
                loc_list.append([int(listx[i])+ 313, int(listy[i])+ 313])
#     loc_list.append([313,313])
    return(loc_list)


def calc_dmap(poten_locs,center,ddash_params):
    ddash_map=[]
    for loc in poten_locs:
#         print('the location in dmape is ',loc)
        loc_deg=(7.5/333)*dist(loc,center)
#         print('the location dist from center is ',loc_deg,' in degrees')
        ddash_map.append(exp_func (loc_deg, ddash_params))
#         print('the ddash for distance ',loc_deg, ' is ',ddash_map[-1])
    return(ddash_map)
        
def visual_search(ddash_params,poten_locs,target_loc_index):

    thr=0.99
    max_nfix=50
    num_points=np.shape(poten_locs)[0]
    fixlist=[]
    fixlist.append([313,313])  #center fixation as the first fixation
    
    prior=np.random.uniform(0,1,num_points)
    posterior=0
    lsum=np.zeros(num_points)
    deltaHs=np.zeros(num_points)

    k=0
    found_flag=False
    while np.max(posterior)<thr and np.shape(fixlist)[0]<max_nfix:
#         print('fixation number is ',k+1)
        curr_loc=fixlist[-1]
#         print('current loc is ',curr_loc)
        d_map=np.array(calc_dmap(poten_locs,curr_loc,ddash_params))
        
        w=-0.5*np.ones(np.shape(poten_locs)[0])
        w[target_loc_index]=0.5
        w_map=np.random.normal(w,1/(d_map),d_map.shape) # temp response at each loc. is random samples from a gaussian distribution with varaince 1/ddash
#         print('w_map',w_map)

        lsum=lsum+d_map*d_map*w_map
#         print('lsum is ',lsum)

        like=np.exp(lsum)
#         print('likelihood is ',like)

        posterior=prior*like
        posterior=posterior/sum(posterior)
#         print('posterior is ',posterior)

        for i in range(0,np.shape(poten_locs)[0]):
            d2=np.array(calc_dmap(poten_locs,poten_locs[i],ddash_params))
            deltaHs[i]=sum(d2*d2*posterior)
#         print('deltaHs is ',deltaHs)
#         print('max deltaHs is ',np.max(deltaHs))

        best_loc=np.unravel_index(np.argmax(deltaHs, axis=None), deltaHs.shape)
#         print('best loc is ',best_loc)

        target_loc=(np.unravel_index(np.argmax(posterior, axis=None),posterior.shape))
        if np.max(posterior)>=thr: #or target_loc_index==target_loc[0]:
            if poten_locs[target_loc_index]==fixlist[-1]:
                print('prob is ', np.max(posterior),' and target found')
                found_flag=True
                fixlist.append(poten_locs[target_loc[0]])
                print('target loc is ',poten_locs[target_loc_index], ' and fixations are ')
                print(fixlist)
            else:
                found_flag=False
            return(fixlist,found_flag)
        else:
            fixlist.append(poten_locs[best_loc[0]])
            found_flag=False
        k=k+1
    print('prob is ', np.max(posterior),' and target found')
    print('target loc is ',poten_locs[target_loc_index], ' and fixations are ')
    print(fixlist)
    return(fixlist,found_flag)

        
        
def get_scanpath_simul(save_dir,iter_num,output_type):
    if output_type=="simul":
        ddash_params=pd.read_csv('files/'+output_type+"_ddash_params.csv")
    elif output_type=="human":
        ddash_params=pd.read_csv('files/'+output_type+"_ddash_params.csv")
    poten_locs=get_loc_tri_arr()
    print('potential target locations are ',poten_locs)

    n_fix_ecc_ha=[] # average number of fixations for all BG and all ecc
#    BG_nums=[1,3,4,5,6,9,10,13,15,17,18,19,20,21,22,24,26,28]
    i=0
    scanpath_lst=[]
    for num in range(0,np.shape(ddash_params)[0]):
        print('')
        print('visual search starts for BG ',ddash_params.iloc[num,0])

#        print(BG_nums.index(BG_num))
        ddash_param=ddash_params.iloc[num,1]
##         print('ddash_params')
        print(ddash_param)
#
        n_fix=[]  # number of fixations for each BG and all ecc at each iteration
        n_fix_ecc=[] # average number of fixations for each BG and all ecc
        for target_loc_index in range(0,np.shape(poten_locs)[0]):  #locations in index
            nfix_ave=0
            target_loc=poten_locs[target_loc_index]
            poten_locs_center=deepcopy(poten_locs)
            ecc_pix=dist(target_loc,[313,313])
            target_ecc_norm=ecc_pix//66+1
            iter_num_counter=0
            while iter_num_counter<iter_num:
                print('try')
                scanpath,found_flag=visual_search(ddash_param,poten_locs_center,target_loc_index)
                if found_flag:
                    iter_num_counter=iter_num_counter+1
                    nfix_ave=nfix_ave+np.shape(scanpath)[0]
                    scanpath_lst.append([ddash_params.iloc[num,0],target_ecc_norm,target_loc,np.shape(scanpath)[0],scanpath])
                    print('nfix for target loc ',target_loc_index,' is in iter ',iter_num-iter_num_counter, ' is ', np.shape(scanpath)[0])
            nfix_ave=nfix_ave/iter_num
#             print('ave nfix is ', nfix_ave)
            print(' which was at ecc ',target_ecc_norm)
#             print(target_ecc_norm)
#             print('real target loc is ',target_loc)
            n_fix.append([target_ecc_norm,nfix_ave])

        n_fix_df=pd.DataFrame(n_fix)

#         print(n_fix_df)
        n_fix_ecc=[ddash_params.iloc[num,0],'mean']
        n_fix_ecc_median=[ddash_params.iloc[num,0],'median']
        n_fix_ecc_std=[ddash_params.iloc[num,0],'std']
        for ecc in [1,2,3,4,5]:
            n_fix_ecc.append(np.array(n_fix_df[n_fix_df.iloc[:,0]==ecc].iloc[:,1]).mean())
            n_fix_ecc_std.append(np.array(n_fix_df[n_fix_df.iloc[:,0]==ecc].iloc[:,1]).std())
            n_fix_ecc_median.append(np.median(np.array(n_fix_df[n_fix_df.iloc[:,0]==ecc].iloc[:,1])))
        n_fix_ecc_ha.append(n_fix_ecc)
        n_fix_ecc_ha.append(n_fix_ecc_std)
        n_fix_ecc_ha.append(n_fix_ecc_median)
        
        t=(np.arange(5)+1)*1.4
        plt.figure()
        plt.plot(t,n_fix_ecc[2:])
        plt.fill_between(t,np.array(n_fix_ecc[2:])+np.array(n_fix_ecc_std[2:]),
                        np.array(n_fix_ecc[2:])-np.array(n_fix_ecc_std[2:]),alpha=0.2)
        plt.title(ddash_params.iloc[num,0])
        plt.ylim(0,30)
        plt.savefig(save_dir+'search_output_'+str(ddash_params.iloc[num,0])+'.png')
        i=i+1
        
    print('n_fix dataframe is ')
    n_fix_ecc_ha_df=pd.DataFrame(n_fix_ecc_ha)
    n_fix_ecc_ha_df.columns=['BG_num','stat','ecc1','ecc2','ecc3','ecc4','ecc5']
    print(n_fix_ecc_ha_df)
    n_fix_ecc_ha_df.to_csv(save_dir+'simul_nfix_'+output_type+'_d_'+str(iter_num)+'iter.csv',index=False)
    
    scanpath_df=pd.DataFrame(scanpath_lst)
    scanpath_df.columns=['BG_num','ecc_norm','target_loc','n_fix','scanpath']
    scanpath_df.to_csv(save_dir+'simul_scanpath_'+output_type+'_d_'+str(iter_num)+'iter.csv',index=False)

    
    
if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser(epilog="Input files read from ./files/*.csv")
    parser.add_argument('--dir', type=str, default='files', help="Output folder")
    parser.add_argument('--iters', type=int, default=1, help="Number of iterations that each search loc and BG is run")
    parser.add_argument('--out_type', choices=['simul', 'human'], help='Type of output required')

    args = parser.parse_args()

    save_dir = args.dir + '/'
    get_scanpath_simul(save_dir, iter_num=args.iters, output_type=args.out_type)  #the argument is iternum
