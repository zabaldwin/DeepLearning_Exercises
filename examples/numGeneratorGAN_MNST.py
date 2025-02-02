#!/usr/bin/env python3

"""
Simple Generative Adversarial Network (GAN) for the MNIST dataset
_________________________________________________________________
Note: This code follows a DCGAN style approach for generating new MNIST like images.
Kept minimal for clarity but can easily tweak hyperparameters
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Conv2D, Conv2DTranspose)
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import multiprocessing as mp

(X_train, _), (_, _) = mnist.load_data()

# Rescale [0, 255] to [-1, 1]
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=-1)  # (N, 28, 28, 1)

img_rows = X_train.shape[1]
img_cols = X_train.shape[2]
channels = X_train.shape[3]

img_shape = (int(X_train.shape[1]), int(X_train.shape[2]), int(X_train.shape[3]))

z_dim = 100

def buildGenerator(z_dim):
    
    model = Sequential()

    model.add(Dense(128 * 7 * 7, input_dim=z_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))  

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(64, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(channels, kernel_size=5, activation='tanh', padding='same'))
    return model


def buildDiscriminator(img_shape):
    
    model = Sequential()

    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=img_shape))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


discriminator = buildDiscriminator((img_shape))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

generator = buildGenerator(z_dim)

# Build the combined GAN model
# z -> generator -> discriminator -> validity
z = Input(shape=(z_dim,))
img = generator(z)

# Freeze discriminator's parameters 
discriminator.trainable = False
validity = discriminator(img)

gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam())

'''
# Iteratively training the discriminator on real & fake samples then tryi and fool the discriminator 
def train_gan(X_train, epochs, batch_size, save_interval):
    
    # 1 = real |  0 = fake
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    d_loss_history = []
    g_loss_history = []

    for epoch in tqdm(range(epochs), desc="Training epochs"):

        #random real images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        #fake images
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(noise)

        # Train on real
        d_loss_real = discriminator.train_on_batch(imgs, real)
        # Train on fake 
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # fool the discriminator to think fake images are real
        g_loss = gan.train_on_batch(noise, real)

        d_loss_history.append(d_loss[0])
        g_loss_history.append(g_loss)

        print(f"Epoch {epoch}, [Discriminator loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%], [Generator loss: {g_loss}]")

        if epoch % save_interval == 0:
            save_images(epoch, generator)

    plot_loss(d_loss_history, g_loss_history)
'''

# Each process has its own variable copy so multiprocessing wont share weights
def train_gan_process(X_data, epochs, batch_size, save_interval):
    
    local_disc = buildDiscriminator((img_shape))
    local_disc.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    local_gen = buildGenerator(z_dim)

    z_input = Input(shape=(z_dim,))
    img_out = local_gen(z_input)
    local_disc.trainable = False
    validity_out = local_disc(img_out)
    local_gan = Model(z_input, validity_out)
    local_gan.compile(loss='binary_crossentropy', optimizer=Adam())

    # 1 = real |  0 = fake
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    d_loss_history = []
    g_loss_history = []

    for epoch in tqdm(range(epochs), desc="Training epochs"):

        #random real images
        idx = np.random.randint(0, X_data.shape[0], batch_size)
        imgs = X_data[idx]

        #random fake images
        noise = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = local_gen.predict(noise)

        # Train on real
        d_loss_real = local_disc.train_on_batch(imgs, real)
        # Train on fake
        d_loss_fake = local_disc.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # fool the discriminator to think fake images are real
        g_loss = local_gan.train_on_batch(noise, real)

        d_loss_history.append(d_loss[0])
        g_loss_history.append(g_loss)

        print(f"Epoch {epoch}, [Discriminator loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%], [Generator loss: {g_loss}]")

        if epoch % save_interval == 0:
            save_images(epoch, local_gen)

    plot_loss(d_loss_history, g_loss_history)


def save_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    
    noise = np.random.normal(0, 1, (examples, z_dim))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_image_epoch_{epoch}.png')
    plt.close()


def plot_loss(d_loss_history, g_loss_history):
    
    plt.figure(figsize=(10, 5))
    plt.plot(d_loss_history, label='Discriminator loss')
    plt.plot(g_loss_history, label='Generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Loss')
    plt.legend()
    plt.savefig('gan_loss_history.png')
    plt.show()


if __name__ == "__main__":
    epochs = 20
    batch_size = 128
    save_interval = 1000

    #train_gan(X_train, epochs, batch_size, save_interval)

    # Sadly just runs things independently through multiple processes for num > 1... could use MirroredStrategy to distribute things but not too helpful using a single CPU
    num_processes = 1

    args_list = [
        (X_train, epochs, batch_size, save_interval)
        for _ in range(num_processes)
    ]

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(train_gan_process, args_list)

    print("All parallel processes have completed")
