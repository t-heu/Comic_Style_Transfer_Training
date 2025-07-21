import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = 256
INPUT = 'test_input.jpg'
OUTPUT = 'test_output.jpg'

def load_img(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)/255.0

def save_img(arr, path):
    img = Image.fromarray((arr*255).astype('uint8'))
    img.save(path)

model = tf.keras.models.load_model('models/ghibli_transform.keras')
img = load_img(INPUT)[np.newaxis]
stylized = model.predict(img)[0]
save_img(stylized, OUTPUT)
print("Resultado salvo:", OUTPUT)
