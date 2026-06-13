import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Charger le modèle entraîné
model = tf.keras.models.load_model("mnist_model.keras")

# Charger les données de test MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisation
x_test = x_test / 255.0

# Ajouter la dimension du canal
x_test = x_test[..., tf.newaxis]

# Choisir une image à tester
index = 0

image = x_test[index]

# Faire la prédiction
prediction = model.predict(np.array([image]))

predicted_digit = np.argmax(prediction)

print("Chiffre réel :", y_test[index])
print("Chiffre prédit :", predicted_digit)

# Afficher l'image
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Prédit : {predicted_digit}")
plt.show()