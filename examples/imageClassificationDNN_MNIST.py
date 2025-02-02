#!/usr/bin/env python3

"""
Simple Neural Network for the MNIST dataset
_______________________________________________________
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import classification_report
from tqdm import tqdm

def prep_data():
    
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Convert [0, 255] to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # print("train_images shape:", train_images.shape)
    # print("test_images shape:", test_images.shape)
    
    return train_images, train_labels, test_images, test_labels


def dnn_model():
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_network(model, train_images, train_labels, epochs=5, batch_size=64, val_split=0.2):
    
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=1  
    )

    return history


def plot_training_metrics(history):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_images, test_labels):
    
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Accuracy of test set: {test_accuracy:.4f}")

    predictions = model.predict(test_images, verbose=0)
    predictions = np.argmax(predictions, axis=-1)
    return predictions


def confuse_matrix(test_labels, predictions):
    
    cm = confusion_matrix(test_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

# Considered 'positive' if it matches that class
def ROC(test_labels, predictions):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(10):
    
        actual_i = (test_labels == i).astype(int)
        predicted_i = (predictions == i).astype(int)

        fpr[i], tpr[i], _ = roc_curve(actual_i, predicted_i)
        
        roc_auc[i] = auc(fpr[i], tpr[i])  

    plt.figure()
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Class")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    
    train_images, train_labels, test_images, test_labels = prep_data()

    model = dnn_model()

    history = train_network(model, train_images, train_labels, epochs=10, batch_size=64, val_split=0.2)

    plot_training_metrics(history)

    predictions = evaluate_model(model, test_images, test_labels)

    confuse_matrix(test_labels, predictions)

    print("\nClassification Report:")
    print(classification_report(test_labels, predictions))

    ROC(test_labels, predictions)
