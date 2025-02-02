#!/usr/bin/env python3

"""
Autoencoder Anomaly Detection on 1D Data
_____________________________________________________
Note; currently this program over generalizes to test data it seems
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import uproot
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

# Assumes TTree named tree and branch named response
def load_data(file_path):
    with uproot.open(file_path) as root_file:
        tree = root_file["tree"]
        arrays = tree.arrays(["response"])
        data = arrays["response"]
    
    return np.array(data)


def prep_data(data):
    
    data_mean = np.mean(data)
    data_std = np.std(data)
    normalized_data = (data - data_mean) / data_std

    normalized_data = normalized_data[:, np.newaxis]

    train_data, val_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
        
    return np.array(train_data), np.array(val_data)


def build_autoencoder(input_shape):
    
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu')
    ])

    decoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_shape, activation='linear')
    ])

    autoencoder = tf.keras.Sequential([
        encoder,
        decoder
    ])

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder


def train_autoencoder(autoencoder, train_data, val_data,
                      epochs=10, batch_size=32):
    
    autoencoder.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(val_data, val_data)
    )


def evaluate_autoencoder(autoencoder, val_data):
    
    reconstructions = autoencoder.predict(val_data)
    reconstruction_errors = np.mean(np.square(val_data - reconstructions), axis=1)
    return reconstruction_errors, reconstructions


if __name__ == "__main__":


    file_path = "normalData.root"
    data = load_data(file_path)
    train_data, val_data = prep_data(data)

    input_shape = train_data.shape[1]  # 1 since data is 1D
    autoencoder = build_autoencoder(input_shape)
    train_autoencoder(autoencoder, train_data, val_data, epochs=10, batch_size=32)

    val_errors, val_recons = evaluate_autoencoder(autoencoder, val_data)

    threshold = np.mean(val_errors) + 3 * np.std(val_errors)
    anomalies_detected = np.sum(val_errors > threshold)
    print(f"Anomalies detected in validation data={threshold:.4f}: {anomalies_detected}")

    val_labels_binary = np.where(val_errors > threshold, 1, 0)
    auc_score = roc_auc_score(val_labels_binary, val_errors)
    print(f"Area under the ROC Curve: {auc_score:.4f}")

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        val_labels_binary, val_labels_binary, average='binary'
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(val_data, label='Original Data')
    plt.plot(val_recons, label='Reconstructed')
    plt.title("Validation: Original vs. Reconstructed")
    plt.xlabel("Index")
    plt.ylabel("Response")
    plt.legend()
    plt.show()

    test_file_path = "mixedData.root"
    test_data = load_data(test_file_path)

    train_mean = np.mean(train_data)
    train_std  = np.std(train_data)
    normalized_test_data = (test_data - train_mean) / train_std
    normalized_test_data = normalized_test_data[:, np.newaxis]

    test_errors, test_recons = evaluate_autoencoder(autoencoder, normalized_test_data)

    test_anomalies = np.sum(test_errors > threshold)
    print(f"Anomalies detected in test data using threshold={threshold:.4f}: {test_anomalies}")

    plt.figure(figsize=(10, 5))
    plt.plot(normalized_test_data, label='Test Data (Normalized)')
    plt.plot(test_recons, label='Reconstructed')
    plt.title("Test: Original vs. Reconstructed")
    plt.xlabel("Index")
    plt.ylabel("Response")
    plt.legend()
    plt.show()
