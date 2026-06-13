import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Charger le modèle
model = tf.keras.models.load_model("mnist_model.keras")

# Charger MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalisation
x_test = x_test / 255.0

# Choisir une image
index = 0
image = x_test[index]

# Ajouter du bruit
noise = np.random.normal(0, 0.2, image.shape)
noisy_image = np.clip(image + noise, 0, 1)

# Préparer pour le CNN
input_image = noisy_image.reshape(1, 28, 28, 1)

# Prédiction
prediction = model.predict(input_image)
predicted_digit = np.argmax(prediction)

print("Chiffre réel :", y_test[index])
print("Chiffre prédit :", predicted_digit)

# Affichage
plt.imshow(noisy_image, cmap="gray")
plt.title(f"Réel : {y_test[index]} | Prédit : {predicted_digit}")
plt.show()