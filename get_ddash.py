import cv2
from PIL import Image
from math import *

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


from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(224),                    #[2]
#  transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


def get_18BGs_detectability(log_fov_ha):
    # A function to fit our simulation output to human experimental data
    # @param log_fov_ha= a list of trained prob.calssifiers to get FA and TP probs from
    
    fov_rsize=expinfo['fov_rsize']
    #defining the Alexnet as the feature extractor
    my_model = models.alexnet(pretrained=True)
    new_classifier = torch.nn.Sequential(*list(my_model.classifier.children())[:-1])
    my_model.classifier = new_classifier
    my_model.eval() #always set Alexnet on eval mdode for test
    
    overlay=cv2.imread(expinfo['overlay_dir'], -1)
    overlay[:,:,:3] = cv2.cvtColor(overlay[:,:,:3], cv2.COLOR_BGR2RGB) # this one has transparency
    
    paths=[img for img in glob.glob(expinfo['test_data_dir'])]
    im_names=[os.path.basename(x) for x in glob.glob(expinfo['test_data_dir'])]
    
    dlist_ha=[]
    counter=0
    for im_paths in paths:  #go over all the images
        image=cv2.cvtColor(cv2.imread(im_paths), cv2.COLOR_BGR2RGB)
        w,h,c=image.shape
        overlay_size=expinfo["overlay_size"]  #usually overlay size should be the same size as DNN input, so
                                              #resizing doesn't cause info loss

        ratio=np.shape(overlay)[0]//np.shape(overlay)[1] #resize overlay to the size of the central foveation patch
        #get ratio to make sure, overlay w/h doesn't change
        overlay_resized=cv2.resize(overlay, (overlay_size-4, (overlay_size-4)*ratio))
        
        dlist=[im_names[counter][:-4]] #define the dlist with im_name as the index
#         plt.figure()
#         plt.subplot(111)
#         plt.imshow(image)
#         plt.title(im_name[-6:-4])
        for fov_num in range(1,6):
            window_size=int(overlay_size*fov_rsize[fov_num-1]) # patch size
            d_ave=[]
            for i in range(0,expinfo['rand_iternum']):
                #to match human experiment with im size 666 and target size 40,
                #resize 666 sized image to 3330 (666/40 *200) so a 200 pixel target
                # so target will have the same size with respect to BG as in human expr
                rand_cut=np.random.randint(0,666-window_size-1)
                im_resized=cv2.resize(image[0:666,0:666,:],(int(16.65*overlay_size),int(16.65*overlay_size)))
                
                neg_patch=im_resized[rand_cut:rand_cut+window_size,rand_cut:rand_cut+window_size]
                neg_fvec=np.array(get_feature_vecs([neg_patch],my_model,expinfo['gray_flag']))
                aug_patch=paste_person([neg_patch],overlay_resized,expinfo['center_flag'])
#                 plt.figure()
#                 plt.subplot(111)
#                 plt.imshow(aug_patch[0])
#                 plt.title(im_name[-6:-4])
                pos_fvec= np.array(get_feature_vecs([aug_patch[0]],my_model,expinfo['gray_flag']))
                y_prob_neg=log_fov_ha[fov_num-1].predict_proba(neg_fvec)
                y_prob_pos=log_fov_ha[fov_num-1].predict_proba(pos_fvec)
                neg_label=log_fov_ha[fov_num-1].predict(neg_fvec)[0]
                pos_label=log_fov_ha[fov_num-1].predict(pos_fvec)[0]
                
                Fa=y_prob_neg[0][1]
                TP=y_prob_pos[0][1]
                dd=norm.ppf(TP)-norm.ppf(Fa)
                if dd<0: #any ddash which is calculated as negative is definitely a hard BG
                    dd=0.000001
                elif dd==inf: #if ddash is inf (easy BG) which means Fa==1 and TP==1, we assign the max ddash
                    dd=16
                d_ave.append(dd)

            dlist.append(np.array(d_ave).mean())
            dlist.append(np.array(d_ave).std())
            
        dlist_ha.append(dlist)
        counter=counter+1
#     print(pd.DataFrame(dlist_ha))
#     pd.DataFrame(dlist_ha3).to_csv('aaa.csv',index=False)
    return(dlist_ha)



### test the calculated d' for the backgrounds used in the human experiment
def analyze_detection_simul(log_fov_ha2,w,human_ddash,h_ddash_datapoints):

    ddash_simul=get_18BGs_detectability(log_fov_ha2)
    ddash_simul_df=pd.DataFrame(ddash_simul)#pd.read_csv('./aaa.csv')
    ddash_simul_df.columns=['im_name','ecc1_ave','ecc1_std','ecc2_ave','ecc2_std'
                ,'ecc3_ave','ecc3_std','ecc4_ave','ecc4_std','ecc5_ave','ecc5_std']
    ddash_simul_df.set_index('im_name', inplace=True)
    print(ddash_simul_df)
    
    t=np.arange(5)*1.8
    dlist=[]
    for i in range(0,np.shape(ddash_simul_df)[0]):
        simul_ddash_lst=np.array(ddash_simul_df.iloc[i,[0,2,4,6,8]])
        bias=simul_ddash_lst[0]-3.28 # b in equation 4 in the NeurIPS paper
        simul_ddash_lst=(simul_ddash_lst-bias)*w #equation 4 in the NeurIPS paper
        popt_exp2, pcov_exp2=curve_fit(exp_func,t[0:4],simul_ddash_lst[0:4])
        if popt_exp2[0]>0.6: #since the range of ddash params are between 0 and 0.6
            popt_exp2[0]=0.6
        if popt_exp2[0]<0: #since the range of ddash params are between 0 and 0.6
            popt_exp2[0]=0
        dlist.append([ddash_simul_df.index[i],popt_exp2[0]])
        print('ddash param for ',ddash_simul_df.index[i],'is',popt_exp2[0])
        
        plt.figure()
        plt.plot(t,exp_func(t,popt_exp2[0]),label='simul',color='g')
        plt.title(ddash_simul_df.index[i])
        plt.ylim([-1,4])
        plt.savefig('files/detectability_output_'+ddash_simul_df.index[i]+'.png')
    dlist_df=pd.DataFrame(dlist)
    dlist_df.columns=['BG','param']
    dlist_df.to_csv('files/simul_ddash_params.csv',index=False)

expinfo={
            "codes_dir":"./",
            "overlay_name":"trans2.png",
            "fov_rsize":[1,1.4,1.8,2.2,2.6],  # the reatio of window size change with diff eccentricities
            "gray_flag":True, # True:make images grayscale and then get fvecs
            "center_flag":True, #True: pastes in the center of BG, Flase, pastes in a random loc. can be [x,y] as a given location to paste
            "DNN":"Alexnet",
            "overlay_size":224,
            "rand_iternum":10 #paste peron in rand_iternum random location and get the average of d'
}
expinfo["overlay_dir"]=expinfo["codes_dir"]+"data/overlays/"+expinfo["overlay_name"]
expinfo["test_data_dir"]=expinfo["codes_dir"]+"data/test/*.png"

with open(expinfo["codes_dir"]+"files/log_fov_ha.pkl",'rb') as file:
    log_fov_ha=pickle.load(file)
    
w=np.load(expinfo["codes_dir"]+"files/w_pretrained.npy") #equation 4 in the NuerIPS paper
human_ddash=pd.read_csv(expinfo["codes_dir"]+"files/human_ddash_mean.csv")
h_ddash_datapoints=pd.read_csv(expinfo["codes_dir"]+"files/human_p_ddash.csv")

analyze_detection_simul(log_fov_ha,w,human_ddash,h_ddash_datapoints)
