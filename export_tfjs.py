
import tensorflowjs as tfjs
import tensorflow as tf

model = tf.keras.models.load_model('models/gibi_model')
tfjs.converters.save_keras_model(model, 'models/gibi_tfjs')
