import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math


from google.colab import drive
drive.mount('/content/drive')


test_image = '/content/drive/MyDrive/detection/'
labels = '/content/drive/MyDrive/labels/'
abd = '/content/drive/MyDrive/test_image/'
file_path = '/content/drive/MyDrive/train_label_segmentation/'


size = 416
for path in os.listdir(labels):  
    with open(labels+path, 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            class_name = line_split[0]
            x3 = float(line_split[3])*416
            x4 = float(line_split[4])*416
            x1 = ((float(line_split[1]))*416)-x3/2
            x2 = ((float(line_split[2]))*416)-x4/2

            with open('/content/train/'+path , 'a') as f:
                f.write(str(class_name) + " " + str(x1) + " " + str(x2) +
                        " " + str(x3) + " " + str(x4) + '\n')


def check(img_name):
    for i in os.listdir(file_path):
        p=i.split('.')
        if(p[0]==img_name):
            return 1
    return 0


for im in os.listdir(abd):
    path = os.path.join(abd, im)
    s = im.split('.')
    img_data = cv2.imread(path)
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    vectorized = np.float32(img.reshape((-1,3)))
    find = check(s[0])
    if(find==1):    
        with open(file_path+s[0]+'.txt', 'r') as f:
            for line in f:
                line_split = line.strip().split(' ')
                class_name=int(line_split[0])
                x = math.ceil(float(line_split[1]))
                y = math.ceil(float(line_split[2]))
                h = math.ceil(float(line_split[3]))
                w = math.ceil(float(line_split[4]))
                
                sub_image = img[y:y+w, x:x+h]
                vectorized = np.float32(sub_image.reshape((-1,3)))
                K = 3
                attempts=10
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts,
                                                cv2.KMEANS_RANDOM_CENTERS)       
                center = np.uint8(center)
                result_image = center[label.flatten()].reshape((sub_image.shape))
                masked_image = np.copy(result_image).reshape((-1, 3)).reshape(result_image.shape)
                
                img[y:y+w, x:x+h] = masked_image
                ims = img     
        cv2.imwrite(os.path.join('/content/segmentation/', im), ims)
    else:
        cv2.imwrite(os.path.join('/content/segmentation', im), img)