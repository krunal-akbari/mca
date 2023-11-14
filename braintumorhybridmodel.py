# -*- coding: utf-8 -*-
"""BrainTumorHybridModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qccPoEaeseUZSFSqfpietS3ENgKkZBvM
"""

#mounting drive
from google.colab import drive
drive.mount('/content/gdrive')

"""#Transfer Learning Model Analysis

##PreProcessing
"""

# Commented out IPython magic to ensure Python compatibility.
# import libraries
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, MaxPool2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import cv2
from tqdm import tqdm
import random
import time
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# %matplotlib inline

#listing directories of training and testing images
train_dir="/content/gdrive/MyDrive/Dataset/train"
test_dir="/content/gdrive/MyDrive/Dataset/test"

#preprocessing and generating training validation and testing dataset
from keras.preprocessing.image import ImageDataGenerator
trainDataGen=ImageDataGenerator(
    rescale=1./255, #rescale
    validation_split=0.1, #validation split
    zoom_range=0.2, #zoom to create augmented sample


)
testDataGen=ImageDataGenerator(rescale=1./255) #rescaling

trainGen=trainDataGen.flow_from_directory(train_dir,
                                          target_size=(224,224), #image size
                                          color_mode='rgb', #color mode of image
                                          class_mode='categorical', #label to be categorized
                                          batch_size=128, #specifying batch size
                                          subset='training'
                                         )
testGen=testDataGen.flow_from_directory(test_dir, target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=1)

valGen=trainDataGen.flow_from_directory(train_dir, target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=128,subset='validation')

"""##VGG16"""

from keras.applications.vgg16 import VGG16

base_model=VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
out=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.inputs,outputs=out)
model.summary()

callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

steps_per_epochs=trainGen.samples//128
print(steps_per_epochs)
validation_steps_=valGen.samples//128
print(validation_steps_)

start = time.time()
history=model.fit(trainGen,validation_data=valGen,epochs=50,callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])), acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluating the model
model_evaluation = model.evaluate(testGen)

print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %") #Calculating accuracy

"""##VGG19"""

from keras.applications.vgg19 import VGG19

base_model=VGG19(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
out=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.inputs,outputs=out)
model.summary()

callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

steps_per_epochs=trainGen.samples//128
print(steps_per_epochs)
validation_steps_=valGen.samples//128
print(validation_steps_)

start = time.time()
history=model.fit(trainGen,validation_data=valGen,epochs=50,callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

#plotting accuracy and loss graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])), acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluating the model
model_evaluation = model.evaluate(testGen)

print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %") #Calculating accuracy

"""##ResNet50"""

from tensorflow.keras.applications import ResNet50

base_model=ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
out=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.inputs,outputs=out)
model.summary()

callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

steps_per_epochs=trainGen.samples//128
print(steps_per_epochs)
validation_steps_=valGen.samples//128
print(validation_steps_)

start = time.time()
history=model.fit(trainGen,validation_data=valGen,epochs=50,callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])), acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluating the model
model_evaluation = model.evaluate(testGen)

print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %") #Calculating accuracy

"""##ResNet101"""

from tensorflow.keras.applications import ResNet101

base_model=ResNet101(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
out=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.inputs,outputs=out)
model.summary()

callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

steps_per_epochs=trainGen.samples//128
print(steps_per_epochs)
validation_steps_=valGen.samples//128
print(validation_steps_)

start = time.time()
history=model.fit(trainGen,validation_data=valGen,epochs=50,callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])), acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluating the model
model_evaluation = model.evaluate(testGen)

print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %") #Calculating accuracy

"""##DenseNet121"""

from tensorflow.keras.applications import DenseNet121

base_model=DenseNet121(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
out=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.inputs,outputs=out)
model.summary()

callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

steps_per_epochs=trainGen.samples//128
print(steps_per_epochs)
validation_steps_=valGen.samples//128
print(validation_steps_)

start = time.time()
history=model.fit(trainGen,validation_data=valGen,epochs=50,callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])), acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluating the model
model_evaluation = model.evaluate(testGen)

print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %") #Calculating accuracy

"""##DenseNet201"""

from keras.applications.densenet import DenseNet201 #importing libraries

#creating model
base_model=DenseNet201(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

x=Flatten()(base_model.output)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.4)(x)
x=Dense(1024,activation='relu')(x)
out=Dense(4,activation='softmax')(x)

model=Model(inputs=base_model.inputs,outputs=out)
model.summary()

#adding early stopping as callback and compiling model
callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

#creating training and validation steps based on batch size
steps_per_epochs=trainGen.samples//128
print(steps_per_epochs)
validation_steps_=valGen.samples//128
print(validation_steps_)

#training model and calculating execution time
start = time.time()
history=model.fit(trainGen,validation_data=valGen,epochs=50,callbacks=callbacks, verbose=1)
end = time.time()
print("Execution time", end-start)

#plotting accuracy and loss graph
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.history['accuracy']))

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(history.history['accuracy'])), acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# evaluatinn the model
model_evaluation = model.evaluate(testGen)

print(f"Model Accuracy:{model_evaluation[1] *100: 0.2f} %") #Calculating accuracy

"""# ClassifierAnalysis

##PreProcessing
"""

train_dir="/content/gdrive/MyDrive/Training"
test_dir="/content/gdrive/MyDrive/Testing"

import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import time
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
trainDataGen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    zoom_range=0.2,


)
testDataGen=ImageDataGenerator(rescale=1./255)

trainGen=trainDataGen.flow_from_directory(train_dir,
                                          target_size=(224,224),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          batch_size=128,
                                          subset='training'
                                         )
testGen=testDataGen.flow_from_directory(test_dir, target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=1)

valGen=trainDataGen.flow_from_directory(train_dir, target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=128,subset='validation')

train_y = []
for i in range(len(trainGen)):
  for j in range(len(trainGen[i])):
    if j == 1:
      for k in trainGen[i][j]:
        train_y.append(k)
print(train_y[0])

test_y = []
for i in range(len(testGen)):
  for j in range(len(testGen[i])):
    if j == 1:
      for k in testGen[i][j]:
        test_y.append(k)
print(test_y[0])

from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D,Input,AveragePooling2D, MaxPool2D
from keras.applications.densenet import DenseNet201

base_model=DenseNet201(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in base_model.layers:
  layer.trainable=False

model = base_model.output
model = MaxPool2D(pool_size=(2,2))(model)
model = Dropout(0.5)(model)
model = Flatten()(model)
model = Model(inputs=base_model.input, outputs=model)
model.summary()

from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks=[EarlyStopping(monitor='val_loss',patience=10)]

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])

x_mod = model.predict(trainGen)

x_mod.shape

y_train_up1 = np.argmax(train_y, axis=-1)

x_testpred = model.predict(testGen)
x_testpred.shape

y_test_up1 = np.argmax(test_y, axis=-1)

"""##SVM"""

from sklearn import svm

svm_lin = svm.SVC(C=1.0, kernel='linear') #specifying linear kernel for SVM
start = time.time()
svm_lin.fit(x_mod, y_train_up1) #fit the model
end = time.time()
print("Execution time:", end-start)
y_pred = svm_lin.predict(x_testpred) #predicting the model
print(classification_report(y_test_up1, y_pred))

print(confusion_matrix(y_test_up1,y_pred)) #confusion matrix

#Heatmap of confusion matrix
fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_up1,y_pred),ax=ax,annot=True,
           linewidths=2, cmap='Blues')
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
             fontname='monospace',y=0.92,x=0.28,)

plt.show()

#calculating accuracy
count = 0
for i in range(len(y_pred)):
  if y_pred[i] == y_test_up1[i]:
    count +=1
print("Accuracy is :", (count/len(y_pred)))

"""##Random Forest"""

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=50, random_state=42) #specifying random forest with 50 trees to be considered for estimation

start = time.time()
RF.fit(x_mod,y_train_up1)
end = time.time()
print("Execution time:", end-start)

pred_RF = RF.predict(x_testpred)

print(classification_report(y_test_up1, pred_RF))

count = 0
for i in range(len(pred_RF)):
  if pred_RF[i] == y_test_up1[i]:
    count +=1
print("Accuracy is :", (count/len(pred_RF)))

"""##KNN"""

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3) #specifying 3 nearest neighbor for knn

start = time.time()
neigh.fit(x_mod,y_train_up1)
end = time.time()
print("Execution time:", end-start)

out_KNN = neigh.predict(x_testpred)

print(classification_report(y_test_up1, out_KNN))

count = 0
for i in range(len(out_KNN)):
  if out_KNN[i] == y_test_up1[i]:
    count +=1
print("Accuracy is :", (count/len(out_KNN)))
