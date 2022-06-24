import numpy as np
import os
import cv2
from random import shuffle

from google.colab import drive
drive.mount('/content/drive')

train_path = '/content/drive/MyDrive/output/'
test_path = '/content/drive/MyDrive/bcc/'
TrainImageLabels = '/content/drive/MyDrive/TrainImageLabels.txt'
TestImageLabels = '/content/drive/MyDrive/TestImageLabels.txt'

Train_data_values = []
file1 = open(TrainImageLabels, "r")
lines = file1.readlines()
lines = lines[1:]
for line in lines:
    p = line.split(" ")
    Train_data_values.append((int(p[0][5:10]),int(p[1]),int(p[2])))
file1.close()

def get_Data(im):
    values = []
    for i in Train_data_values:
      if im == i[0]:
        values.append(i[1])
        values.append(i[2])
    return values

IMG_SIZE = 416
def create_train_data():
    training_data = []
    for img in os.listdir(train_path):
        im = int(img[5:10])
        path = os.path.join(train_path, img)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        Values = get_Data(im)
        training_data.append([(np.array(img_data)), Values[0], Values[1]])
    shuffle(training_data)
    return training_data

train_data = create_train_data()

import tensorflow as tf
from tensorflow.keras.layers import  Dense,Flatten,Dropout
from keras.models import Sequential

train = train_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE,3)
y_train = np.array([i[2] for i in train])

y_t =[]
for i in y_train:
  label = np.zeros(2)
  if i == 1:
    label[0] = 1
  else:
    label[1] =1
  y_t.append(label)

from keras.applications  import inception_resnet_v2
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import SGD
from keras.applications import resnet

baseModel = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(416, 416, 3)))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['accuracy'])

model.fit(X_train, np.array(y_t), validation_split=0.2, epochs=15)

Test_data_values = []
file1 = open(TestImageLabels, "r")
lines = file1.readlines()
lines = lines[1:]
for line in lines:
    p = line.split(" ")
    Test_data_values.append((int(p[0][5:10]),int(p[1]),int(p[2])))
file1.close()

def get_Data(im):
    values = []
    for i in Test_data_values:
      if im == i[0]:
        values.append(i[1])
        values.append(i[2])
    return values

def create_test_data():
    test_data = []
    for img in os.listdir(test_path):
        im = int(img[5:10])
        path = os.path.join(test_path, img)
        img_data = cv2.imread(path)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        Values = get_Data(im)
        test_data.append([(np.array(img_data)), Values[0], Values[1]])
    shuffle(test_data)
    return test_data

test_data = create_test_data()

test = test_data
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE,3)
y_test = np.array([i[2] for i in test])
print(np.shape(X_test))
print(np.shape(y_test))

y =[]
for i in y_test:
  label = np.zeros(2)
  if i == 1:
    label[0] = 1
  else:
    label[1] =1
  y.append(label) 
print(np.shape(y))
print(y)
print(y_test)

score = model.evaluate(X_test, np.array(y), verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])