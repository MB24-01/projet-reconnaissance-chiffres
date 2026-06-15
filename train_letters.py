import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Charger EMNIST Letters
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

# Prétraitement des images EMNIST
def preprocess(image, label):
    # Correction orientation EMNIST
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)

    # Normalisation entre 0 et 1
    image = tf.cast(image, tf.float32) / 255.0

    # Labels EMNIST : 1 à 26 -> 0 à 25
    label = label - 1

    return image, label

ds_train = ds_train.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# Architecture CNN pour les lettres
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(26, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=3,
    restore_best_weights=True
)

model.fit(
    ds_train,
    epochs=30,
    validation_data=ds_test,
    callbacks=[early_stop]
)

loss, accuracy = model.evaluate(ds_test)
print(f"Précision lettres : {accuracy*100:.2f}%")

model.save("letters_model.keras")