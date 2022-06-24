##install yelo5
!git clone  'https://github.com/ultralytics/yolov5.git'
!pip install -r './yolov5/requirements.txt'


##libraries
import os
import cv2
import shutil 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##connect to grive
from google.colab import drive
drive.mount('/content/drive')


##paths
train_path =  '/content/drive/MyDrive/new_ image/'
test_path = '/content/drive/MyDrive/test_image/'
real_image = '/content/drive/MyDrive/Train/Images/'
real_image_test='/content/drive/MyDrive/Test/Images'
Annotations = '/content/drive/MyDrive/Train/Annotations/'


#label: jpg, png
def get_label(img_name):
    for i in os.listdir(real_image_test):
        p = i.split('.')
        if(p[0]==img_name):
           return p[1]


#train, validation
counter = 0
for i in os.listdir(test_path):
     im = cv2.imread(test_path+i)
     im = cv2.resize(im,(416,416),interpolation=cv2.INTER_LINEAR)
     p = i.split(".")
     tmp = get_label(p[0])
     path_Annotations = Annotations+p[0] + '.' + tmp + '.txt'
     if(counter<2700):    
        shutil.copy(path_Annotations,'/content/new/')
        cv2.imwrite(os.path.join('/content/new_test_images/' + p[0] + '.' + tmp), im)
     else:
        cv2.imwrite(os.path.join('/content/File/images/valid/', p[0] + '.jpg'), im)
        shutil.copy(path_Annotations, '/content/newv')
    counter=counter+1


##Normalization
def change(x, y, w, h, img):
    x = ((x+int((w)/2))/img[1])
    w = (w/img[1])
    y = ((y+int((h)/2))/img[0])
    h = (h/img[0])
    return x,y,w,h


##incoder
col =['Bicycle', 'Boat','Bottle', 'Bus', 'Car', 'Cat','Chair', 'Cup', 'Dog', 'Motorbike','People','Table']
from sklearn import preprocessing
def Feature_Encoder(col) :
    label = preprocessing.LabelEncoder()
    label.fit(list(col))
    col = label.transform(list(col)) 
    return col
r = Feature_Encoder(col)


##read annotation then create labels and normalize data
train_label = '/content/new/'
for path in os.listdir(train_label):
    with open(train_label+path, 'r') as f:
        s = path.split(".")
        tmp = get_label(s[0])
        img = cv2.imread(real_image + '/' + s[0] + '.' + tmp)
        x = img.shape[0]
        y = img.shape[1]  
        count = 1
        for line in f:
            if count != 1:
                line_split = line.strip().split(' ')
                class_name = line_split[0]
                x1 = line_split[1]
                x2 = line_split[2]
                x3 = line_split[3]
                x4 = line_split[4]
                d = 0  
                for i in col:
                  if(class_name==i):
                    t = r[d]
                    break
                  d = d+1 
                x1, x2, x3, x4 = change(int(x1), int(x2), int(x3), int(x4), [x,y])
                with open('/content/File/labels/train/' + s[0] + '.txt', 'a') as f:
                    f.write(str(t) + " " + str(x1) + " " + str(x2) + " " + str(x3) 
                            + " " + str(x4) + '\n')
            count+=1


##read annotation then create labels and normalize data (validation)
valid_label='/content/newv/'
for path in os.listdir(valid_label):
    with open(valid_label+path, 'r') as f:
        s=path.split(".")
        tmp= get_label(s[0])
        img=cv2.imread(real_image+'/'+s[0]+'.'+tmp)
        x=img.shape[0]
        y=img.shape[1]  
        count = 1
        for line in f:
            if count != 1:
                line_split = line.strip().split(' ')
                class_name = line_split[0]
                x1 = line_split[1]
                x2 = line_split[2]
                x3 = line_split[3]
                x4 = line_split[4]
                x1, x2, x3, x4 = change(int(x1),int(x2),int(x3),int(x4),[x,y])
                with open('/content/File/labels/valid/'+s[0]+'.txt' , 'a') as f:
                    f.write(str(t)+" "+str(x1)+" "+str(x2)+" "+str(x3)+" "+str(x4)+'\n')
            count+=1


##finding files txt
# def check_file(name):
#     for i in os.listdir('/content/drive/MyDrive/File/images/train/'):
#        p=i.split('.')
#        if(p[0]==name):
#          return 1
#        else:
#          return p[0] 
     

# for path in os.listdir('/content/drive/MyDrive/File/labels/train/'):
#     s=path.split('.')
#     found=check_file(s[0])
#     if(found==1):
#       continue
#     else:
#       print(found)
#       break
            
            
# with open(train_label+path, 'r') as f:
#       tmp= get_label(found)
#       img=cv2.imread(real_image+'/'+found+'.'+tmp)
#       x=img.shape[0]
#       y=img.shape[1]  
#       count = 1
#       for line in f:
#           if count != 1:
#               line_split = line.strip().split(' ')
#               class_name = line_split[0]
#               x1 = line_split[1]
#               x2 = line_split[2]
#               x3 = line_split[3]
#               x4 = line_split[4]
#               d=0
#               for i in col:
#                 if(class_name==i):
#                   t=r[d]
#                   break
#                 d=d+1 
#               x1, x2, x3, x4 = change(int(x1),int(x2),int(x3),int(x4),[x,y])
#               with open('/content/fi/'+found+'.txt', 'a') as f:
#                   f.write(str(t)+" "+str(x1)+" "+str(x2)+" "+str(x3)+" "+str(x4)+'\n')
#           count+=1
#       print('/content/File/labels/train/'+found+'.txt')

# shutil.copy('/content/File/fi/'+found+'.txt','/content/File/labels/train/')


##create file yaml (train path, validation path, labels)
!echo -e 'train: /content/drive/MyDrive/File/images/train/\nval: /content/drive/MyDrive/File/images/valid/\n\nnc: 12\nnames: ['Bicycle', 'Boat','Bottle', 'Bus', 'Car', 'Cat','Chair', 'Cup', 'Dog', 'Motorbike','People','Table']' >> ahmed.yaml


##train model
!python /content/yolov5/train.py --img 640 --batch 16  --epochs 25 \
    --data /content/ahmed.yaml --cfg /content/yolov5/models/yolov5x.yaml  --weights yolov5x.pt --name {detection_20}

!python /content/yolov5/detect.py  --save-txt  --imgsz 640 --data '/content/ahmed.yaml' --source '/content/drive/MyDrive/bcc' --weights '/content/yolov5/runs/train/{detection_20}2/weights/best.pt'


##save file
fil = ['ImageName','nBicycle','nBoat','nBottle','nBus','nCar','nCat','nChair','nCup','nDog','nMotorbike','nPeople','nTable']

df = pd.DataFrame(fil)  

def labb(na):
    if na=='Bicycle,' or na=='Bicycles,':
        return 0
    elif na=='Boat,' or na=='Boats,' :
        return 1
    elif na=='Bottle,' or na=='Bottles,':
        return 2
    elif na=='Bus,' or na=='Buss,' :
        return 3
    elif na=='Car,' or na=='Cars,':
        return 4
    elif na=='Cat,' or na=='Cats,':
        return 5
    elif na=='Chair,' or na=='Chairs,':
        return 6
    elif na=='Cup,' or na=='Cups,':
        return 7
    elif na=='Dog,' or na=='Dogs,' :
        return 8
    elif na=='Motorbike,' or na=='Motorbikes,':
        return 9
    elif na=='People,' or na=='Peoples,':
        return 10
    elif na=='Table,' or na=='Tables,':
        return 11
    else:
        return 'no'

with open('/content/5.txt') as f:
    contents = f.readlines()
    
d=[]
cou=0
for i in range(len(contents)):
    data = contents[i].split(' ')
    da = data[2].split('/')
    name = da[-1][:-1]
    objc = data[4:-2]
    arr = np.zeros(12)
    cou+=1
    
    n = []
    for j in range(len(objc)):
        if len(objc[j])>2:
            arr[labb(objc[j])]=objc[j-1]
    
    n.append(name)
    for h in range(12) :
        n.append(arr[h])
    
    d.append(n)

pd.DataFrame(daaa).to_csv('/content/drive/MyDrive/ha.csv', index=False,header=fil)