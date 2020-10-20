import pickle
from copy import deepcopy
#
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec

import cv2
from PIL import Image

import torch
import torchvision
from torchvision import models

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#

from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(224),                    #[2]
#  transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


def get_feature_vecs(augmented_images,model,gray_flag):
    # The function to produce the feature vectors input images of through Alexnet
    # inputs RGB images (changes them to Grayscale), and the modified alexnet
    # outputs the features from last layer of the Alexnet as vectors with size 4096
    # @ param augmented_images: list of m*n*3 matrices (RGB images)
    # @ param model: the shortened alexnet model
    # @ param gray_flag: boolean variable, if True gets features from a grayscale image
    # @ param feature_vecs: array of 4096 elements
    
    feature_vecs=[]
    for i in range(0,np.shape(augmented_images)[0]):
        if gray_flag==False:
            feature_vecs.append(model(torch.unsqueeze(transform(Image.fromarray(augmented_images[i])),0)).data.numpy()[0])
        elif gray_flag==True:
            w,h,c=augmented_images[i].shape
            im=np.zeros((w,h,c),dtype=np.uint8)
            # trying grayscale images for the human exoeriment
            im[:,:,0]=deepcopy(cv2.cvtColor(augmented_images[i],cv2.COLOR_BGR2GRAY))
            im[:,:,1]=deepcopy(cv2.cvtColor(augmented_images[i],cv2.COLOR_BGR2GRAY))
            im[:,:,2]=deepcopy(cv2.cvtColor(augmented_images[i],cv2.COLOR_BGR2GRAY))
            feature_vecs.append(model(torch.unsqueeze(transform(Image.fromarray(im)),0)).data.numpy()[0])

    return(feature_vecs)


def makepatches(imarr,windowsize,windowstep):
    # makes patches with a specific size from the given images
    # @param imarr: one image
    # @param windowsize: integer, widowe size of the patches as
    # @param windows: list of patches
    
    windows=[]
    windowsize_r = windowsize
    windowsize_c = windowsize
    # Crop out the window and calculate the histogram
    for r in range(0,imarr.shape[0]-windowsize_r+1 , windowstep):
        for c in range(0,imarr.shape[1]-windowsize_c+1 , windowstep):
            windows.append(imarr[r:r+windowsize_r,c:c+windowsize_c])
#     print("the shape of patches is "+str(np.shape(window)))
    return(windows)


def paste_person(windows,overlay,center_flag):
    # @ param windows: list of n*n*3 np array, BG patch on which overlay is pasted
    # @ param overlay: m*m*4 np array, transparent overlay
    # @ param center_flag: boolean, if True, paste on center of patch, if False, paste on a random x,y

    augmented_images=[]
    for i in range(0,np.shape(windows)[0]):
        if center_flag==True:
            x=(np.shape(windows[0])[0]-np.shape(overlay)[0])//2
            y=(np.shape(windows[0])[1]-np.shape(overlay)[1])//2
        elif center_flag==False:
            x=np.random.randint(1,np.shape(windows[0])[0]-np.shape(overlay)[0]+1)
            y=np.random.randint(1,np.shape(windows[0])[1]-np.shape(overlay)[1]+1)
            if x>np.shape(windows[0])[0]-np.shape(overlay)[0]:
                x=np.shape(windows[0])[0]-np.shape(overlay)[0]
            if y>np.shape(windows[0])[1]-np.shape(overlay)[1]:
                y=np.shape(windows[0])[1]-np.shape(overlay)[1]
        if type(center_flag) is list:
            x=center_flag[0]
            y=center_flag[1]
            
        h, w, c = windows[i].shape
        wall=np.zeros((h,w,4))
        wall[x:x+np.shape(overlay)[0], y:y+np.shape(overlay)[1]] = overlay

        result = np.zeros((h, w, 3), np.uint8)

        alpha = wall[:, :, 3] / 255.0
        result[:, :, 0] = (1. - alpha) * windows[i][:, :, 0] + alpha * wall[:, :, 0]
        result[:, :, 1] = (1. - alpha) * windows[i][:, :, 1] + alpha * wall[:, :, 1]
        result[:, :, 2] = (1. - alpha) * windows[i][:, :, 2] + alpha * wall[:, :, 2]

        augmented_images.append(result)
    return(augmented_images)


def my_log(df,fov_num):
    # all functions needed for replacing Alexnet with a real human to find detectibility maps
    # gets the dataframe of neg and pos features and trains the log regression
    # oututs conf matrix and trained log
    # @ param df:
    arr = np.shape(df)[0]
    out = np.random.RandomState(seed=42).permutation(arr) # random shuffle
    for i in range(0,arr):
        df[i,:]=deepcopy(df[out[i],:])

    ##randomly select 80% of the instances to be training and the rest to be testing
    X_train, X_test, y_train, y_test = train_test_split(df[:,0:-1],df[:,-1], train_size=0.8, test_size=0.20, random_state=42)

    log = LogisticRegression(random_state=0, solver='liblinear')
    log.fit(X_train, y_train)
    
    y_pred = log.predict(X_test)
    p_pred=log.predict_proba(X_test)
    print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
    print('accuracy is: '+ str(((X_test.shape[0]-(y_test != y_pred).sum())/X_test.shape[0])*100))
    np.save(expinfo['output_save_load_dir']+'Ypred_'+expinfo['BG_folder']+'_'+expinfo['classifier']+'_'+expinfo['DNN']+'_fov'+str(fov_num),y_pred)
    np.save(expinfo['output_save_load_dir']+'Ppred_'+expinfo['BG_folder']+'_'+expinfo['classifier']+'_'+expinfo['DNN']+'_fov'+str(fov_num),p_pred)
    return(log)


def plot_classifier_output_hist():
    plt.figure(figsize=(20, 8))
    gs=gridspec.GridSpec(1,5)
    for fov_num in range(1,6):
        ax=plt.subplot(gs[fov_num-1])
        p_pred=np.load(expinfo['output_save_load_dir']+'Ppred_'+expinfo['BG_folder']+'_'+expinfo['classifier']+'_'+expinfo['DNN']+'_fov'+str(fov_num)+'.npy')
        plt.hist(p_pred[:,1],bins=20)
        
        plt.title('Prob. hist of featurs for log'+str(fov_num))
        
        
def exp_func (x, b):
    return (3.28*np.exp(-b*x))

