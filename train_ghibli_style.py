import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
from PIL import Image
import os

# === Configurações ===
CONTENT_DIR = 'datasets/content'
STYLE_IMAGE = 'datasets/style/style.png'
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50
CONTENT_WEIGHT = 1.0
STYLE_WEIGHT = 10.0

def preprocess_vgg(x):
    x = x * 255.0  # converte de [0,1] para [0,255]
    return preprocess_input(x)

# === Data Loading ===
def load_img(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img) / 255.0

def dataset_generator():
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [os.path.join(CONTENT_DIR, f) for f in os.listdir(CONTENT_DIR)]
    for path in files:
        # Ignora arquivos que não são imagens (ex: .gitkeep)
        if not path.lower().endswith(valid_exts):
            print(f"Ignorando arquivo não imagem: {path}")
            continue
        
        img = load_img(path)
        yield img

dataset = tf.data.Dataset.from_generator(
    dataset_generator,
    output_types=tf.float32,
    output_shapes=(IMG_SIZE, IMG_SIZE, 3)
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Modelo Transformer ===
def transformer():
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 9, padding='same', activation='relu')(inp)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(3, 9, padding='same', activation='sigmoid')(x)
    return Model(inp, x)

# === Funções auxiliares ===
def gram_matrix(x):
    x = tf.transpose(x, perm=[0, 3, 1, 2])  # b, c, h, w
    b = tf.shape(x)[0]
    c = tf.shape(x)[1]
    h = tf.shape(x)[2]
    w = tf.shape(x)[3]
    features = tf.reshape(x, (b, c, h * w))
    gram = tf.matmul(features, features, transpose_b=True) / tf.cast(h * w * c, tf.float32)
    return gram

# === Extrator VGG19 ===
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1']
content_layer = 'block4_conv2'
outputs = [vgg.get_layer(l).output for l in style_layers + [content_layer]]
vgg_model = Model(vgg.input, outputs)

# Pré-compute estilo
style_img = load_img(STYLE_IMAGE)[np.newaxis]
style_outputs = vgg_model(preprocess_vgg(style_img))
style_targets = [gram_matrix(s) for s in style_outputs[:len(style_layers)]]

# === Treinamento ===
transform_net = transformer()
opt = tf.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(content_batch):
    with tf.GradientTape() as tape:
        out = transform_net(content_batch)                        # gera a imagem estilizada (0..1)
        
        # Pré-processa antes de enviar pro VGG (espera imagens no formato correto)
        out_vgg = vgg_model(preprocess_vgg(out))                 
        style_out = out_vgg[:len(style_layers)]
        content_out = out_vgg[len(style_layers):]
        
        # Extrai features do conteúdo original também pré-processado
        content_features = vgg_model(preprocess_vgg(content_batch))[-1]

        # Calcula as perdas usando gram_matrix corrigida
        style_loss = tf.add_n([
            tf.reduce_mean(tf.square(gram_matrix(o) - t)) 
            for o, t in zip(style_out, style_targets)
        ])
        
        content_loss = tf.reduce_mean(tf.square(content_out[0] - content_features))
        
        # Aplica os pesos ajustados
        loss = STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss

    grads = tape.gradient(loss, transform_net.trainable_variables)
    opt.apply_gradients(zip(grads, transform_net.trainable_variables))
    return loss

for epoch in range(EPOCHS):
    for batch in dataset:
        loss = train_step(batch)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.numpy():.4f}")

transform_net.save('models/ghibli_transform.keras')
print("Modelo salvo: models/ghibli_transform.keras")
