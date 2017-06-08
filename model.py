from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Cropping2D
import csv, os
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
import  sys
import argparse


import matplotlib.pyplot as plt
from keras import backend as K



__author__ = 'Curt'

print('Python Version:', sys.version)

from sklearn.model_selection import train_test_split


def nvidia_model(input_shape):
    model = Sequential()

    # Crop image to critical area.
    model.add(Cropping2D(cropping=((60, 28), (30, 30)), input_shape=input_shape))

    # Normalization
    model.add(Lambda(lambda x: (x / 127.5) - 1.0))

    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))
    return model


class Generator:
    """A generator class"""

    def __init__(self, path):
        self.data_file_path = path

        pts1 = np.float32([[60, 60], [250, 60], [60, 130]])

        # Setup transformation matrix for shear left
        pts2 = np.float32([[30, 60], [220, 60], [60, 130]])
        self.M_left = cv2.getAffineTransform(pts1, pts2)

        # Setup transformation matrix for shear right
        pts2 = np.float32([[90, 60], [280, 60], [60, 130]])
        self.M_right = cv2.getAffineTransform(pts1, pts2)

    def shear_image(self, image, direction):
        if direction == 'left':
            M = self.M_left
        elif direction == 'right':
            M = self.M_right
        else:
            raise RuntimeError('Invalid shear direction')
        rows, cols, ch = image.shape
        sheared_image = cv2.warpAffine(image, M, (cols, rows))

        # plt.figure()
        # plt.imshow(image, interpolation='nearest')
        #
        # plt.figure()
        # plt.imshow(sheared_image, interpolation='nearest')

        return sheared_image

    def generator(self, samples, batch_size=35):
        num_samples = len(samples)
        batch_size = int(batch_size / 5)

        while 1: # Loop forever so the generator never terminates
            samples = shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:

                    center_file_name = batch_sample[0].split('\\')[-1]
                    center_path = os.path.join(self.data_file_path, 'IMG', center_file_name)
                    center_image = cv2.imread(center_path)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)

                    # augment the data by simulating car pointing to left
                    veer_right_image = self.shear_image(center_image, direction='right')
                    vr_angle = center_angle + 0.20
                    images.append(veer_right_image)
                    angles.append(vr_angle)

                    # augment the data by simulating car pointing to right
                    veer_left_image = self.shear_image(center_image, direction='right')
                    vl_angle = center_angle - 0.20
                    images.append(veer_left_image)
                    angles.append(vl_angle)

                    # try to train the car to remain in the center by steering to the right if the car is
                    # on the left
                    left_file_name = batch_sample[1].split('\\')[-1]
                    left_path = os.path.join(self.data_file_path, 'IMG', left_file_name)
                    left_image = cv2.imread(left_path)
                    left_angle = center_angle + 0.20
                    images.append(left_image)
                    angles.append(left_angle)

                    # try to train the car to remain in the center by steering to the left if the car is
                    # on the right
                    right_file_name = batch_sample[2].split('\\')[-1]
                    right_path = os.path.join(self.data_file_path, 'IMG', right_file_name)
                    right_image = cv2.imread(right_path)
                    right_angle = center_angle - 0.2
                    images.append(right_image)
                    angles.append(right_angle)


                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)


def read_data(data_file_path):
    lines = []

    # read inputs
    csv_file_path = os.path.join(data_file_path, 'driving_log.csv')
    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

        return lines


def get_layer_outputs(model, test_image):
    """
    awesome code visualize layers copied from 
    http://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer/41712013#41712013
    """
    outputs    = [layer.output for layer in model.layers]          # all layer outputs
    comp_graph = [K.function([model.input]+ [K.learning_phase()], [output]) for output in outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs


def plot_layer_outputs(model, test_image, layer_number):
    layer_outputs = get_layer_outputs(model, test_image)

    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n     = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]

    for img in L:
        plt.figure()
        plt.imshow(img, cmap='gray', interpolation='nearest')


def main():
    data_path = os.path.join('.', 'data-track-1')
    # data_path = os.path.join('.', 'data')
    print('Path: ', data_path)

    samples = read_data(data_path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    gen = Generator(data_path)

    batch_size = 150
    train_generator = gen.generator(train_samples, batch_size=batch_size)
    validation_generator = gen.generator(validation_samples, batch_size=batch_size)

    input_shape = (160, 320, 3)
    model = nvidia_model(input_shape)

    model.compile( optimizer='adam', loss='mean_squared_error')

    center_file_name = train_samples[0][0]
    center_image = cv2.imread(center_file_name )

    # pts1 = np.float32([[60, 60], [250, 60], [60, 130]])
    # # Setup transformation matrix for shear left
    # pts2 = np.float32([[30, 60], [220, 60], [60, 130]])
    # M_left = cv2.getAffineTransform(pts1, pts2)
    # # Setup transformation matrix for shear right
    # pts2 = np.float32([[90, 60], [280, 60], [60, 130]])
    # M_right = cv2.getAffineTransform(pts1, pts2)
    # rows, cols, ch = center_image.shape
    # sheared_image = cv2.warpAffine(center_image, M_right, (cols, rows))
    #
    # plt.imshow(center_image)
    # plt.figure()
    # plot_layer_outputs(model, [sheared_image], 0)
    # plt.show()

    model.fit_generator(train_generator, steps_per_epoch=len(train_samples)*5/batch_size, validation_data = validation_generator, nb_val_samples = len(validation_samples)*5/batch_size, nb_epoch = 6)
    # model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*3, validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 5)
    model.save('model.h5')


if __name__ == "__main__":
    sys.exit(main())