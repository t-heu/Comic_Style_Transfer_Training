
import tensorflow as tf
from tensorflow.keras import layers
import os
from PIL import Image
import numpy as np

CONTENT_DIR = 'datasets/content'
STYLE_DIR = 'datasets/style'
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100  # Aumente depois

def load_image(path):
    img = Image.open(path).resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return img

def load_dataset(dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f != '.gitkeep']
    imgs = [load_image(f) for f in files]
    return tf.convert_to_tensor(imgs, dtype=tf.float32)

def build_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (9, 9), strides=1, padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), strides=2, padding='same'),
        layers.Activation('relu'),
        layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same'),
        layers.Activation('relu'),
        layers.Conv2D(3, (9, 9), strides=1, padding='same'),
        layers.Activation('sigmoid')
    ])
    return model

def train():
    content_images = load_dataset(CONTENT_DIR)
    style_images = load_dataset(STYLE_DIR)

    model = build_model()
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(content_images, style_images, epochs=EPOCHS, batch_size=BATCH_SIZE)
    model.save('models/gibi_model.keras')

if __name__ == "__main__":
    train()
