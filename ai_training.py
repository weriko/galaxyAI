import h5py
import numpy as np
from tensorflow.keras import utils
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
import time
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

start = time.time()

#From astroNN documentation
with h5py.File('Galaxy10.h5', 'r') as f:
    images = np.array(f['images'])
    labels = np.array(f['ans'])


labels = utils.to_categorical(labels, 10)


labels = labels.astype(np.float32)
images = images.astype(np.float32)
print(labels[0])
shape = images[0].shape
print(shape)
model = Sequential()
model.add(Conv2D(1024, (3,3), input_shape = shape  ))
model.add(Conv2D(512, (3,3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(256,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(Conv2D(128,(3,3)))
model.add(GlobalMaxPooling2D())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()

model.fit(images,labels,epochs=2)
model.save(f'model{str(start)}.h5')