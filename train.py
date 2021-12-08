import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers

batch_size = 32
img_height = 32
img_width = 32

# Load the training and validation dataset
train_ds = keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size
)

# Cache the images to improve performences
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model
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

# Compile it
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# We can load the model to continue training the current saved model
# model = keras.models.load_model("model")

# Train the model for 10 epochs
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    batch_size=batch_size,
)
# Save the trained model
model.save("model")


loss, acc = model.evaluate(val_ds, verbose=2)
print(f'Loss: {loss:.2%} Accuracy: {acc:.2%}')
