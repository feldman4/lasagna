from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from lasagna.imports import *
from keras.preprocessing.image import ImageDataGenerator
import keras.backend

keras.backend.set_image_data_format('channels_first')

input_shape = (3, 51, 51)

def load_data():
    pad_zero = lambda x: np.append(x, np.zeros_like(x)[:,:1], axis=1)
    X_pos = read('examples/10X_positive-sgRNA-all_B2.phenotype_aligned.tif')
    X_neg = read('examples/10X_nontargeting-sgRNA-all_B2.phenotype_aligned.tif')

    X = pad_zero(np.concatenate((X_pos, X_neg)))
    y = np.concatenate((np.ones (len(X_pos), dtype=bool),
                        np.zeros(len(X_neg), dtype=bool)))

    return X, y


def make_training_datagen():
    return ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./15000,
        horizontal_flip=True,
        fill_mode='nearest'
    )  

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model