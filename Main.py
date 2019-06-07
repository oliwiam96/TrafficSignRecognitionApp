import csv
import os
import numpy as np
import imageio
import keras
from PIL import Image
from cv2 import cv2
from keras import Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten, BatchNormalization
from numpy import array
from tflearn.data_utils import shuffle

images = []
masks = []
k = 10
X = []
Y = []
circles = 0
crosses = 0
empty = 0
x_train = []
x_test = []
y_train = []
y_test = []
Xtest = []
Ytest = []
PHOTO_SAMPLES = 1000
image_size = 88
# ładuj zdjęcia
directory = os.getcwd() + "\Train" + '\\'
for folder in os.listdir("Train"):
    folderConent = os.listdir("Train\\" + folder)
    for file in folderConent:
        pngfile =cv2.imread(directory + folder + "\\" + file)
       # image_from_array = pngfile
        # The images are of different size
        # this quick hack resizes them to the same size

        size_image = cv2.resize(pngfile,(image_size,image_size))
        x_train.append(array(size_image)/255)
        y_train.append(int(folder))

directory = os.getcwd() + "\Test" + '\\'
folderConent = os.listdir("Test\\")
for file in folderConent:
    pngfile = cv2.imread(directory + "\\" + file)
    image_from_array = Image.fromarray(pngfile, 'RGB')
    size_image = image_from_array.resize((image_size, image_size))

    x_test.append(array(size_image)/255)
with open('Test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            y_test.append(int(row[6]))


num_classes = 43
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = np.array(x_train)
x_test = np.array(x_test)
# x_train, y_train = shuffle(x_train, y_train)
# x_test, y_test = shuffle(x_test, y_test)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
model = Sequential()

model.add(
    Conv2D(50,5,input_shape=(image_size, image_size, 3),strides=(1,1),
           padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(128,3,strides=(1,1),
           padding='valid',activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(
    Conv2D(256,3,strides=(1,1),
           padding='valid',activation='relu')
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(
    Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))
model.add(
    Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

adam = optimizers.adam(lr=0.00005)
# optimizer sgd lub adam
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,batch_size=64,
                    validation_data=(x_test, y_test),shuffle=True,
                epochs=10,verbose=1)
model.save("newModel8.hdf5")
# history = model.fit(Xtrain, Ytrain,batch_size=128,
#                     validation_data=(Xtest, Ytest),
#                 epochs=50,verbose=1)