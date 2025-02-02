#!/usr/bin/env python3

"""
Simple feedforward neural network for the Iris dataset
______________________________________________________
Note: cross-entropy is usually recommended for classification problems
but using a mean squared error (MSE) for simplicity
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm

def prep_data(test_split=0.2, random_seed=42):
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # print("Initial shape X:", X.shape)
    # print("Unique labels:", np.unique(y))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse_output=False)  
    y_onehot = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=test_split, random_state=random_seed
    )
    return X_train, X_test, y_train, y_test

# Less stable gradients in this small of a network if using ReLU or tanh
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(sigmoid_output):
    return sigmoid_output * (1.0 - sigmoid_output)


def initialize(n_input, n_hidden, n_output, seed=42):
    
    np.random.seed(seed)
    W_hidden = np.random.randn(n_input, n_hidden) * 0.01
    b_hidden = np.zeros(n_hidden)
    W_output = np.random.randn(n_hidden, n_output) * 0.01
    b_output = np.zeros(n_output)

    return W_hidden, b_hidden, W_output, b_output


def forward_prop(X, W_hidden, b_hidden, W_output, b_output):
    
    hidden_pre = np.dot(X, W_hidden) + b_hidden
    hidden_act = sigmoid(hidden_pre)

    output_pre = np.dot(hidden_act, W_output) + b_output
    output_act = sigmoid(output_pre)  # Could consider a softmax here

    return hidden_pre, hidden_act, output_pre, output_act


def backward_prop(X, y, hidden_pre, hidden_act, output_pre, output_act,
                       W_hidden, W_output, b_hidden, b_output, lr):
    
    error_output = output_act - y
    delta_output = error_output * sigmoid_prime(output_act)

    error_hidden = np.dot(delta_output, W_output.T)
    delta_hidden = error_hidden * sigmoid_prime(hidden_act)

    W_output -= lr * np.dot(hidden_act.T, delta_output)
    b_output -= lr * np.sum(delta_output, axis=0)

    W_hidden -= lr * np.dot(X.T, delta_hidden)
    b_hidden -= lr * np.sum(delta_hidden, axis=0)

    return W_hidden, b_hidden, W_output, b_output


def accuracy(X, y, W_hidden, b_hidden, W_output, b_output):
    
    _, _, _, predictions = forward_prop(X, W_hidden, b_hidden, W_output, b_output)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y, axis=1)
    
    return np.mean(pred_labels == true_labels)


def train_network(X_train, y_train, n_input, n_hidden, n_output,
                  epochs=200, lr=0.1):
    
    W_hidden, b_hidden, W_output, b_output = initialize(n_input, n_hidden, n_output)

    loss_history = []
    accuracy_history = []

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        
        h_pre, h_act, o_pre, o_act = forward_prop(X_train, W_hidden, b_hidden, W_output, b_output)

        # MSE loss
        loss = np.mean((y_train - o_act)**2)
        loss_history.append(loss)

        acc = accuracy(X_train, y_train, W_hidden, b_hidden, W_output, b_output)
        accuracy_history.append(acc)

        W_hidden, b_hidden, W_output, b_output = backward_prop(
            X_train, y_train, h_pre, h_act, o_pre, o_act,
            W_hidden, W_output, b_hidden, b_output, lr
        )

        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch:03d}: Loss={loss:.4f}, Accuracy={acc:.4f}")

    return W_hidden, b_hidden, W_output, b_output, loss_history, accuracy_history


def plot_training_results(losses, accuracies):
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, c='blue', label='Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(accuracies, c='red', label='Accuracy')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def confuse_matrix(X_test, y_test,
                             W_hidden, b_hidden, W_output, b_output):
    
    _, _, _, test_preds = forward_prop(X_test, W_hidden, b_hidden, W_output, b_output)
    predicted = np.argmax(test_preds, axis=1)
    actual = np.argmax(y_test, axis=1)

    cm = confusion_matrix(actual, predicted)
    iris = load_iris()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = prep_data()

    # Can set more than 5 hidden neurons ... somewhat arbitrary
    n_features = X_train.shape[1]
    n_hidden_neurons = 5
    n_classes = y_train.shape[1]

    # Train 
    W_hidden, b_hidden, W_output, b_output, loss_rec, acc_rec = train_network(
        X_train, y_train, n_features, n_hidden_neurons, n_classes,
        epochs=200, lr=0.1
    )

    final_train_acc = accuracy(X_train, y_train, W_hidden, b_hidden, W_output, b_output)
    print(f"\nFinal Training Accuracy: {final_train_acc:.3f}")

    plot_training_results(loss_rec, acc_rec)

    confuse_matrix(X_test, y_test, W_hidden, b_hidden, W_output, b_output)

    print(classification_report(np.argmax(y_test, axis=1), np.argmax(forward_prop(X_test, W_hidden, b_hidden, W_output, b_output)[-1], axis=1)))
