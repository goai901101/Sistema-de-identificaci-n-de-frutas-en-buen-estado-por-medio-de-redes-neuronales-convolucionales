import sys
import numpy as np
import pandas as pd
import os.path
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report
import os
import zipfile
import glob, cv2, numpy as np
from tqdm import tqdm
import h5py
# example of zoom image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator


categ = ['fresh_apple', 'fresh_banana', 'fresh_orange', 'fresh_tomato', 'fresh_capsicum',
         'stale_apple', 'stale_banana', 'stale_orange', 'stale_tomato', 'stale_capsicum',
         'others'
         ]

X = []
Y = []
path_or = "D:/Practicas/V2/imagenes"
for categ in tqdm(categ):
  
  print("***********"+categ+"***********" )
  # 1
  # 1
  if categ == 'fresh_apple' or categ == 'fresh_banana' or categ == 'fresh_orange' or categ == 'stale_apple' or categ == 'stale_banana' or categ == 'stale_orange':
    pth_img = glob.glob( path_or + '/'+categ+'/*.png')
    
    for pth_img in pth_img: 
        print("Numero")
        print(len(X))       
        img = cv2.resize(cv2.imread(pth_img), (150,150))
        X.append(img)
        if categ == 'fresh_apple':
          Y.append(0)
        elif categ == 'fresh_banana' :
          Y.append(1)
        elif categ == 'fresh_orange' :
          Y.append(2)
        elif categ == 'stale_apple' :
          Y.append(5)
        elif categ == 'stale_banana' :
          Y.append(6)
        elif categ == 'stale_orange':
          Y.append(7)  
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        ##################################################################
        datagen1 = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it1 = datagen1.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen2 = ImageDataGenerator(height_shift_range=0.5)
        # prepare iterator
        it2 = datagen2.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen3 = ImageDataGenerator(horizontal_flip=True)
        # prepare iterator
        it3 = datagen3.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen4 = ImageDataGenerator(rotation_range=90)
        # prepare iterator
        it4 = datagen4.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen5 = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it5 = datagen5.flow(samples, batch_size=1)
        #   Random Zoom Augmentation
        # create image data augmentation generator
        datagen6 = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it6 = datagen6.flow(samples, batch_size=1)
        for i in range(2):
          if categ == 'fresh_apple':
            Y.append(0)
          elif categ == 'fresh_banana' :
            Y.append(1)
          elif categ == 'fresh_orange' :
            Y.append(2)
          elif categ == 'stale_apple' :
            Y.append(5)
          elif categ == 'stale_banana' :
            Y.append(6)
          elif categ == 'stale_orange':
            Y.append(7)  
          # generate batch of images
          batch1 = it1.next()
          # convert to unsigned integers for viewing
          X.append(batch1[0].astype('uint8'))
          # generate batch of images
          batch2 = it2.next()
          # convert to unsigned integers for viewing
          X.append( batch2[0].astype('uint8'))
          # generate batch of images
          batch3 = it3.next()
          # convert to unsigned integers for viewing
          X.append(batch3[0].astype('uint8'))
          # generate batch of images
          batch4 = it4.next()
          # convert to unsigned integers for viewing
          X.append( batch4[0].astype('uint8') )
          # generate batch of images
          batch5 = it5.next()
          # convert to unsigned integers for viewing
          X.append(batch5[0].astype('uint8'))
          # generate batch of images
          batch6 = it6.next()
          # convert to unsigned integers for viewing
          X.append(batch6[0].astype('uint8'))


   # 4
  elif categ == 'fresh_tomato' or categ == 'fresh_capsicum' or categ == 'stale_tomato' or categ == 'stale_capsicum' or categ == 'others':    
    pth_img = glob.glob(path_or + '/'+categ+'/*.jpg')  
    for pth_img in pth_img:    
        print("Numero")
        print(len(X))    
        img = cv2.resize(cv2.imread(pth_img), (150,150))
        X.append(img)
        if categ == 'fresh_tomato': 
          Y.append(3)
        elif categ == 'fresh_capsicum': 
          Y.append(4)
        elif categ == 'stale_tomato':
          Y.append(8)
        elif categ == 'stale_capsicum' :
          Y.append(9)
        elif categ == 'others':    
          Y.append(10)
        # convert to numpy array
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data, 0)
        # create image data augmentation generator
        ##################################################################
        datagen1 = ImageDataGenerator(width_shift_range=[-200,200])
        # prepare iterator
        it1 = datagen1.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen2 = ImageDataGenerator(height_shift_range=0.5)
        # prepare iterator
        it2 = datagen2.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen3 = ImageDataGenerator(horizontal_flip=True)
        # prepare iterator
        it3 = datagen3.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen4 = ImageDataGenerator(rotation_range=90)
        # prepare iterator
        it4 = datagen4.flow(samples, batch_size=1)
        # create image data augmentation generator
        datagen5 = ImageDataGenerator(brightness_range=[0.2,1.0])
        # prepare iterator
        it5 = datagen5.flow(samples, batch_size=1)
        #   Random Zoom Augmentation
        # create image data augmentation generator
        datagen6 = ImageDataGenerator(zoom_range=[0.5,1.0])
        # prepare iterator
        it6 = datagen6.flow(samples, batch_size=1)
        for i in range(2):
          if categ == 'fresh_tomato': 
            Y.append(3)
          elif categ == 'fresh_capsicum': 
            Y.append(4)
          elif categ == 'stale_tomato':
            Y.append(8)
          elif categ == 'stale_capsicum' :
            Y.append(9)
          elif categ == 'others':    
            Y.append(10)
          # generate batch of images
          batch1 = it1.next()
          # convert to unsigned integers for viewing
          X.append(batch1[0].astype('uint8'))
          # generate batch of images
          batch2 = it2.next()
          # convert to unsigned integers for viewing
          X.append( batch2[0].astype('uint8'))
          # generate batch of images
          batch3 = it3.next()
          # convert to unsigned integers for viewing
          X.append(batch3[0].astype('uint8'))
          # generate batch of images
          batch4 = it4.next()
          # convert to unsigned integers for viewing
          X.append( batch4[0].astype('uint8') )
          # generate batch of images
          batch5 = it5.next()
          # convert to unsigned integers for viewing
          X.append(batch5[0].astype('uint8'))
          # generate batch of images
          batch6 = it6.next()
          # convert to unsigned integers for viewing
          X.append(batch6[0].astype('uint8'))
  

X = np.array(X).astype('uint8')
Y = np.expand_dims(np.array(Y).astype('uint8'), axis = 1)

print(len(X))
print(len(Y))

from sklearn.utils import shuffle
X, Y = shuffle(X, Y)
print(Y)
name_mod = 'data_aug'
path5 = "D:/Practicas/V2/"
print("********** h5py **********") 
import h5py
with h5py.File( path5 +name_mod +'.hdf5','w') as hf:
  x = hf.create_dataset('X', data = X, shape = X.shape, compression = 'gzip', compression_opts = 9, chunks = True )
  y = hf.create_dataset('Y', data = Y, shape = Y.shape, compression = 'gzip', compression_opts = 9, chunks = True )
with h5py.File( path5 + name_mod +'.hdf5','r') as hf:
  X = hf['X'][:]
  Y = hf['Y'][:]

Y = tf.keras.utils.to_categorical(Y)
X = X.astype('float32')

X /= 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation= 'relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(11)
                                    
])

from tensorflow.keras.optimizers import RMSprop


print("********** Compile **********") 
model.compile(optimizer=RMSprop(learning_rate = 1e-4),
  loss = 'binary_crossentropy',
  metrics = ['acc'])


# 20 % de Total
x_test = X[int(((len(X)*80 )/100)):]
y_test = Y[int(((len(Y)*80 )/100)):]
# 75 % de Total
x_train, y_train = X[:int(((len(X)*75 )/100))], Y[:int(((len(Y)*75 )/100))]
epocas = 10

print("********** History **********") 
history = model.fit(x_train, y_train, epochs= epocas, batch_size=64, validation_data = (x_test, y_test), verbose = 1)
print(history)


print("********** Results **********") 
results = model.evaluate(x_test, y_test, verbose=0)
print(results)
model.save(path5 + name_mod +'.h5')

y_pred = np.argmax(model.predict(x_test),axis = 1)
cm = confusion_matrix(np.argmax(y_test, axis = 1 ), y_pred)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks=[.5,1.5, 2.5 , 3.5 , 4.5, 5.5 ,6.5,7.5,8.5,9.5,10.5], labels=
        ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Fresh Tomato', 'Fresh Capsicum',
         'Stale Apple', 'Stale Banana', 'Stale Orange', 'Stale Tomato', 'Stale Capsicum',
         'Other'
         ])
plt.yticks(ticks=[.5,1.5, 2.5 , 3.5 , 4.5, 5.5 ,6.5,7.5,8.5,9.5,10.5], labels=
        ['Fresh Apple', 'Fresh Banana', 'Fresh Orange', 'Fresh Tomato', 'Fresh Capsicum',
         'Stale Apple', 'Stale Banana', 'Stale Orange', 'Stale Tomato', 'Stale Capsicum',
         'Other'
         ], rotation=0)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

