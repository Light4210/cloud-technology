import keras
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Reshape, InputLayer, Flatten

def conv_net(classes):

    model = models.Sequential()
    model.add(InputLayer([224, 224, 3]))
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dense(64, activation='relu'))
    model.add(Flatten())
    model.add(Dense(classes, activation='sigmoid'))

    return model