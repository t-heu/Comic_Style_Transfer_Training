
import tensorflow as tf
import numpy as np
from PIL import Image
import os

IMG_SIZE = 256
INPUT_IMAGE = "test_input.jpg"
OUTPUT_IMAGE = "test_output.jpg"
MODEL_PATH = "models/gibi_model.keras"  # ou "models/gibi_model.h5"

def load_and_preprocess_image(path):
    img = Image.open(path).resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # [1, H, W, 3]
    return img

def save_image(img_array, path):
    img_array = np.clip(img_array[0], 0, 1) * 255
    img = Image.fromarray(img_array.astype("uint8"))
    img.save(path)

def run():
    model = tf.keras.models.load_model(MODEL_PATH)
    input_image = load_and_preprocess_image(INPUT_IMAGE)
    output_image = model.predict(input_image)
    save_image(output_image, OUTPUT_IMAGE)
    print(f"Resultado salvo em {OUTPUT_IMAGE}")

if __name__ == "__main__":
    run()
