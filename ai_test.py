import PIL
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
from tensorflow import one_hot
import json
from tensorflow_addons.layers import Sparsemax
import pickle
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model = tf.keras.models.load_model('model1605722831.449136.h5')

def show_img(x):
    
    plt.imshow(x)
    plt.show()
    

with h5py.File('Galaxy10.h5', 'r') as F:
    images = np.array(F['images'])
    labels = np.array(F['ans'])
test = np.expand_dims(images[1253],0)

print(model.predict([test]))
print(labels[1253])
show_img(images[1253])
