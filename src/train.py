import os
import sys
import yaml
import numpy as np
from tensorflow.keras import utils # convert to one-hot-encoding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

def build_model():
    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))    

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def get_callbacks(min_lr):
    return [ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=min_lr), ]

def get_data_generator(train_X):
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    datagen.fit(train_X)
    return datagen

def fit(model, datagen, data, epochs, batch_size, callbacks):
    history = model.fit_generator(datagen.flow(data["train_X"], data["train_y"], batch_size=batch_size),
                                  epochs = epochs, validation_data = (data["valid_X"], data["valid_y"]),
                                  verbose = 2, steps_per_epoch=data["train_X"].shape[0] // batch_size,
                                  callbacks=callbacks)
    return history

def prepare_data(path):
    input_train_x = os.path.join(path, 'train_X.npy')
    input_train_y = os.path.join(path, 'train_y.npy')

    input_valid_x = os.path.join(path, 'valid_X.npy')
    input_valid_y = os.path.join(path, 'valid_y.npy')

    train_X = np.load(input_train_x)
    train_y = np.load(input_train_y)
    train_y = utils.to_categorical(train_y, num_classes = 10)

    valid_X = np.load(input_valid_x)
    valid_y = np.load(input_valid_y)  
    valid_y = utils.to_categorical(valid_y, num_classes = 10)  

    return {
        "train_X": train_X,
        "train_y": train_y,
        "valid_X": valid_X,
        "valid_y": valid_y
    }

input_path = sys.argv[1]
output_path = sys.argv[2]

params = yaml.safe_load(open('params.yaml'))['train']
epochs = params['epochs']
batch_size = params['batch_size']
min_lr = params['min_lr']

os.makedirs(os.path.join('data', 'model'), exist_ok=True)

model = build_model()
callbacks = get_callbacks(min_lr)

data = prepare_data(input_path)
datagen = get_data_generator(data['train_X'])
history = fit(model, datagen, data, epochs, batch_size, callbacks)
model.save(output_path)