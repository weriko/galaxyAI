import PIL
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
from tensorflow import one_hot
import json
from tensorflow_addons.layers import Sparsemax
import pickle
import tensorflow as tf
import h5py
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model = tf.keras.models.load_model('model1605721915.9856515.h5')


with h5py.File('Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])
test = np.expand_dims(images[100],0)

print(model.predict([test]))
print(labels[100])
