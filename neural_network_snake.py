# Disable TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pycodestyle
import warnings

import keras.models
from keras.optimizers import Adam
from keras.layers.core import Dense  # , Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers.convolutional import Conv2D
# from keras.layers.pooling import MaxPooling2D


class NeuralNetwork1:
    """ Neural Network for solving Minesweeper at difficulty 1 (i.e. size 8x8) """

    def __init__(self, learning_rate, epochs, save_file=None):
        # Dimensions of input and output
        self.size_states = 12
        self.size_actions = 3  # straight, left-turn, right-turn

        # Standard file where the NN weights are saved
        self.standard_file = 'test_1'

        # Set up the model
        self.epochs = epochs
        self.alpha = learning_rate
        self.model = self.define_model()

        # Copy the model to create a target model
        self.target_model = keras.models.clone_model(self.model)
        self.update_target_model()

        # Load weights?
        if save_file:
            self.load_NN(save_file)

    def define_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Start with an input layer
            model = keras.models.Sequential()
            model.add(Dense(output_dim=100, activation='relu', input_dim=self.size_states))

            # Add the dense hidden layers
            model.add(Dense(output_dim=100, activation='relu'))

            # Add an 'output' layer (which is just another dense layer)
            model.add(Dense(output_dim=self.size_actions, activation='softmax'))

            # Configure the model and return it
            model.compile(optimizer=Adam(self.alpha), loss='mse', metrics=['accuracy'])

            return model

    def predict_one(self, state):
        """ State (input) is a regular 1D numpy array """
        return self.model.predict(np.array([state]))[0]

    def target_predict_one(self, state):
        """ State (input) is a regular 1D numpy array """
        return self.target_model.predict(np.array([state]))[0]

    def train_one(self, state, desired_output):
        """ State (input) is a regular 1D numpy array, desired_output is a regular 1D numpy array """
        self.model.fit(np.array([state]), np.array([desired_output]), epochs=self.epochs, verbose=0)

    def train_batch(self, list_of_states, list_of_outputs):
        """ Input: list of 2D numpy arrays, output: list of 1D numpy arrays"""
        self.model.fit(np.array(list_of_states), np.array(list_of_outputs), epochs=self.epochs, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_NN(self, file=None):
        # Determine where the weights are saved
        save_file = file if file else self.standard_file
        save_file = "saved_NNs/" + save_file

        # Save the weights
        self.model.save_weights(save_file)

    def load_NN(self, file=None):
        # Determine where the weights are stored
        load_file = file if file else self.standard_file
        load_file = "saved_NNs/" + load_file

        # Load the weights
        self.model.load_weights(load_file)


class Memory:
    def __init__(self, max_memory_size=50000):
        self.max_memory = max_memory_size
        self.states = []
        self.qsas = []

    def add_state(self, state):
        self.states.append(state)

        if len(self.states) > self.max_memory:
            self.states.pop(0)

    def add_list_of_states(self, list_of_states):
        self.states += list_of_states

        if len(self.states) > self.max_memory:
            self.states = self.states[(len(self.states) - self.max_memory):]

    def add_qsa(self, q_s_a):
        self.qsas.append(q_s_a)

        if len(self.qsas) > self.max_memory:
            self.qsas.pop(0)

    def add_list_of_qsas(self, list_of_qsas):
        self.qsas += list_of_qsas

        if len(self.qsas) > self.max_memory:
            self.qsas = self.qsas[(len(self.qsas) - self.max_memory):]

    def get_batch(self, num_samples):
        if num_samples > len(self.states):
            return self.states, self.qsas
        else:
            indices = np.random.choice(len(self.states), num_samples)
            return np.array(self.states)[indices], np.array(self.qsas)[indices]


if __name__ == '__main__':
    pycodestyle.StyleGuide().check_files([__file__])
