import sys
import cv2
import os

from google.colab import drive
drive.mount('/content/drive')

#train_path = '/content/drive/MyDrive/new_ image/'
test_path = '/content/drive/MyDrive/Test/Images/'
#TrainImageLabels = '/content/drive/MyDrive/TrainImageLabels.txt'
#TestImageLabels = '/content/drive/MyDrive/TestImageLabels.txt'


for i in os.listdir(test_path):
     im=cv2.imread(test_path+'/'+i)
     im=cv2.resize(im,(416,416),interpolation=cv2.INTER_LINEAR)
     p = i.split(".")
     cv2.imwrite(os.path.join('/content/image/',p[0]+'.jpg'),im)

path ='/content/Zero-DCE_extension/Zero-DCE++/lowlight_test.py'
!python  -tt /content/Zero-DCE_extension/Zero-DCE++/lowlight_test.py

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    scale_factor = 12
    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight)/255.0)


    data_lowlight = torch.from_numpy(data_lowlight).float()

    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool(scale_factor).cuda()
    DCE_net.load_state_dict(torch.load('/content/Zero-DCE_extension/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth'))
    start = time.time()
    enhanced_image,params_maps = DCE_net(data_lowlight)

    end_time = (time.time() - start)

    print(end_time)
    image_path = image_path.replace('/content/image/','/content/new_ image/')

    result_path = image_path
    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
      os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
    # import pdb;pdb.set_trace()
    torchvision.utils.save_image(enhanced_image, result_path)
    return end_time

if __name__ == '_main_':
    with torch.no_grad():
      filePath = '/content/image/'	
      file_list = os.listdir(filePath)
      
      sum_time = 0
      test_list=[]
      
      for file_name in file_list:
        test_list.append(filePath+file_name) 
      
        for image in test_list:

        print(image)
        sum_time = sum_time + lowlight(image)

      print(sum_time)