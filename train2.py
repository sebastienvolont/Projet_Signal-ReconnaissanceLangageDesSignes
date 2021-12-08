from pprint import pprint
from string import ascii_lowercase

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import img_to_array, load_img

batch_size = 32
img_height = 32
img_width = 32

train_ds = keras.utils.image_dataset_from_directory(
    "img2",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    "img2",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size
)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential([
    layers.Rescaling(1 / 255),
    layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26)
])

# model = Sequential([
#     layers.Rescaling(1 / 255),
#     layers.Flatten(input_shape=(256, 256)),
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(26)
# ])

model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# cp_callback = ModelCheckpoint(
#     filepath='data/cp-{epoch:04d}.ckpt',
#     save_weights_only=True,
#     save_freq=1300,
#     verbose=1
# )
model.load_weights("data/cp.ckpt")

# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=50,
#     batch_size=batch_size,
#     # callbacks=[cp_callback]
# )
# model.save_weights("data/cp.ckpt")

loss, acc = model.evaluate(val_ds, verbose=2)

print(loss, acc)

img = np.array([img_to_array(load_img('img2/b/b.000b0a80-56a3-11ec-8997-e4aaea784a6e.jpg', color_mode='grayscale'))])
pprint(model.predict(img))

img = np.array([img_to_array(load_img('test2/b.jpg', color_mode='grayscale'))])
print('Prediction for image b:', ascii_lowercase[np.argmax(model.predict(img))])
# pprint(model.predict(img))

img = np.array([img_to_array(load_img('test2/c.jpg', color_mode='grayscale'))])
print('Prediction for image c:', ascii_lowercase[np.argmax(model.predict(img))])
# pprint(model.predict(img))

img = np.array([img_to_array(load_img('test2/k.jpg', color_mode='grayscale'))])
print('Prediction for image k:', ascii_lowercase[np.argmax(model.predict(img))])
# pprint(model.predict(img))
