import tensorflow as tf
from tensorflow.keras import layers, models

# Charger MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisation
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ajouter le canal (noir/blanc)
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Construire le CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# Évaluation
loss, accuracy = model.evaluate(x_test, y_test)

print(f"Précision finale : {accuracy*100:.2f}%")

# Sauvegarde
model.save("mnist_model.keras")