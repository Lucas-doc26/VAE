import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from keras import ops
from keras import layers
import cv2

def preprocess_images(target_shape, csv):
    imagens = []
    df = pd.read_csv(csv)
    for path in df["caminho_imagem"]:
        img = cv2.imread(path)  # carrega como BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte para RGB se quiser
        imagens.append(img)

    X_processed = []
    for img in imagens:
        img_resized = cv2.resize(img, target_shape)
        X_processed.append(img_resized)
    #X_processed = np.expand_dims(X_processed, -1)  # (N, 28, 28, 1)
    return np.array(X_processed).astype("float32") / 255.0

treino = preprocess_images((64,64), "CSV/PUC/PUC_Segmentado_Treino.csv")
teste = preprocess_images((64,64), "CSV/PUC/PUC_Segmentado_Validacao.csv")

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

latent_dim = 32

encoder_inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16 * 16 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 64))(x)
x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

mnist_digits = np.concatenate([treino, teste], axis=0)
#mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=150, batch_size=16)

import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import tensorflow.image as tf_img

def calcular_ssim(image1, image2):
    return tf_img.ssim(image1, image2, max_val=1.0).numpy()

def plot_autoencoder(x_test, Autoencoder, width=64, height=64, caminho_para_salvar=None):
    def normalize(image):
        image = np.clip(image, 0, 1)  # Garante que a imagem esteja no intervalo [0, 1]
        return (image - image.min()) / (image.max() - image.min()) if image.max() != image.min() else image

    plt.figure(figsize=(16, 8))

    avaliacoes = []
    for i in range(8):
        # Imagem original
        plt.subplot(2, 8, i + 1)
        plt.imshow(x_test[i])
        plt.title("Original")
        plt.axis("off")

        # Predição e normalização
        z_mean, z_log_var, z = Autoencoder.encoder(x_test[i].reshape((1, width, height, 3)))
        pred = Autoencoder.decoder(z)
        pred_img = normalize(pred[0])

        plt.subplot(2, 8, i + 8 + 1)
        plt.imshow(pred_img)

        ssim = float(calcular_ssim(pred, pred_img))
        avaliacoes.append(ssim)

        del pred_img, pred
        plt.title(f"SSIM: {ssim:.2f}")
        plt.axis("off")

    plt.show()
    media_ssim = np.mean(avaliacoes)
    
    if caminho_para_salvar != None:
        save_path = os.path.join(caminho_para_salvar, 'Autoencoder.png')
        plt.savefig(save_path)

        arquivo = os.path.join(caminho_para_salvar,'media_ssim.txt')
        with open(arquivo, 'w') as f:
            for av in avaliacoes:
                f.write(f'{av}\n')
            f.write(f'Media geral: {media_ssim}')
    
    plt.close("all") 

plot_autoencoder(teste, vae, caminho_para_salvar='/home/lucas/VAE')