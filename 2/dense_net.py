import keras
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, InputLayer, Flatten

def dense_net(classes):
    
    model = models.Sequential()
    model.add(InputLayer([224, 224, 3]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Flatten())
    model.add(Dense(classes, activation='softmax'))

    return model