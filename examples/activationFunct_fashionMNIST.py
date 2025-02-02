#!/usr/bin/env python3

"""
Neural Network for Fashion MNIST
____________________________________________________
Note: Multiple activation functions ReLU, Sigmoid, Tanh, LeakyReLU, ELU
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU, ELU
from tensorflow.keras.utils import to_categorical


def prep_data():
    
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    
    X_train = X_train / 255.0
    X_test  = X_test / 255.0
    
    y_train = to_categorical(Y_train, num_classes=10)
    y_test  = to_categorical(Y_test, num_classes=10)
    
    return X_train, X_test, y_train, y_test

def createModel(activation):
        
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'), 
    ])
    
    if isinstance(activation, str):
        model.add(Dense(128, activation=activation))
    else:
        model.add(Dense(128))
        model.add(activation())

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_various_activations(X_train, y_train, activation_functions, epochs=10):
    
    histories = {}
    for name, activation in activation_functions.items():
        print(f"Training model with {name} activation function...")

        model = createModel(activation=activation)

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=0)
        histories[name] = history
    
    return histories


def best_activation(histories):
    
    best_activation = None
    highest_val_accuracy = 0.0

    for name, history in histories.items():
        val_accuracy = max(history.history['val_accuracy'])  
        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            best_activation = name
    
    return best_activation, highest_val_accuracy


def plot_histories(histories):
    
    plt.figure(figsize=(14, 8))
    for name, history in histories.items():
        plt.plot(history.history['accuracy'], label=f'{name} Training')
        plt.plot(history.history['val_accuracy'], '--', label=f'{name} Validation')

    plt.title('Training and Validation Accuracy by Activation Function')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prep_data()

    activation_functions = {
        'ReLU': 'relu',
        'Sigmoid': 'sigmoid',
        'Tanh': 'tanh',
        'LeakyReLU': LeakyReLU,
        'ELU': ELU
    }

    histories = train_various_activations(X_train, y_train, activation_functions, epochs=10)

    best_activation, highest_val_accuracy = best_activation(histories)
    print(f"\nThe best activation function is '{best_activation}' with a validation accuracy of {highest_val_accuracy:.4f}")

    plot_histories(histories)
