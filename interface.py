import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

# Charger les deux modèles
model_digits = tf.keras.models.load_model("mnist_model.keras")
model_letters = tf.keras.models.load_model("letters_model.keras")

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Reconnaissance de chiffres et lettres par IA")
app.geometry("1000x750")
app.resizable(False, False)

canvas_width = 360
canvas_height = 360

image = Image.new("L", (canvas_width, canvas_height), 0)
draw = ImageDraw.Draw(image)

def dessiner(event):
    x, y = event.x, event.y
    r = 16
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

def preparer_image(img):
    bbox = img.getbbox()

    if bbox is None:
        return Image.new("L", (28, 28), 0)

    img = img.crop(bbox)
    w, h = img.size

    if w > h:
        new_w = 20
        new_h = int(h * 20 / w)
    else:
        new_h = 20
        new_w = int(w * 20 / h)

    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    new_img.paste(img, (left, top))

    return new_img

def reconnaitre():
    img = preparer_image(image)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prédiction chiffre
    pred_digit = model_digits.predict(img_array, verbose=0)
    digit = int(np.argmax(pred_digit))
    conf_digit = float(np.max(pred_digit) * 100)

    # Prédiction lettre
    pred_letter = model_letters.predict(img_array, verbose=0)
    letter_index = int(np.argmax(pred_letter))
    letter = letters[letter_index]
    conf_letter = float(np.max(pred_letter) * 100)

    img.save("dernier_test.png")

    # Comparaison des confiances
    if conf_digit >= conf_letter:
        resultat = str(digit)
        confiance = conf_digit
        message = "Chiffre reconnu"
    else:
        resultat = letter
        confiance = conf_letter
        message = "Lettre reconnue"

    label_resultat.configure(text=resultat)
    label_message.configure(text=message)
    label_confiance.configure(text=f"Confiance : {confiance:.2f}%")

def effacer():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=0)
    label_resultat.configure(text="-")
    label_message.configure(text="Dessinez un chiffre ou une lettre")
    label_confiance.configure(text="Confiance : -")

header = ctk.CTkFrame(app, height=90, corner_radius=0)
header.pack(fill="x")

title = ctk.CTkLabel(
    header,
    text="Reconnaissance automatique de caractères manuscrits",
    font=("Arial", 28, "bold")
)
title.pack(pady=(18, 5))

subtitle = ctk.CTkLabel(
    header,
    text="Projet IA - CNN entraînés sur MNIST et EMNIST Letters",
    font=("Arial", 15)
)
subtitle.pack()

main = ctk.CTkFrame(app, corner_radius=20)
main.pack(padx=25, pady=25, fill="both", expand=True)

left = ctk.CTkFrame(main, width=430, corner_radius=18)
left.pack(side="left", padx=25, pady=25, fill="y")

ctk.CTkLabel(
    left,
    text="Zone de dessin",
    font=("Arial", 22, "bold")
).pack(pady=(20, 10))

canvas = tk.Canvas(
    left,
    width=canvas_width,
    height=canvas_height,
    bg="black",
    highlightthickness=3,
    highlightbackground="#1F6AA5"
)
canvas.pack(padx=20, pady=10)
canvas.bind("<B1-Motion>", dessiner)

button_row = ctk.CTkFrame(left, fg_color="transparent")
button_row.pack(pady=20)

btn_rec = ctk.CTkButton(
    button_row,
    text="Reconnaître",
    command=reconnaitre,
    width=160,
    height=45,
    font=("Arial", 17, "bold")
)
btn_rec.pack(side="left", padx=10)

btn_clear = ctk.CTkButton(
    button_row,
    text="Effacer",
    command=effacer,
    width=130,
    height=45,
    fg_color="#C0392B",
    hover_color="#922B21",
    font=("Arial", 17, "bold")
)
btn_clear.pack(side="left", padx=10)

right = ctk.CTkFrame(main, corner_radius=18)
right.pack(side="right", padx=25, pady=25, fill="both", expand=True)

ctk.CTkLabel(
    right,
    text="Résultat",
    font=("Arial", 26, "bold")
).pack(pady=(35, 10))

label_resultat = ctk.CTkLabel(
    right,
    text="-",
    font=("Arial", 120, "bold")
)
label_resultat.pack(pady=5)

label_message = ctk.CTkLabel(
    right,
    text="Dessinez un chiffre ou une lettre",
    font=("Arial", 20)
)
label_message.pack(pady=10)

label_confiance = ctk.CTkLabel(
    right,
    text="Confiance : -",
    font=("Arial", 20, "bold")
)
label_confiance.pack(pady=10)

info = ctk.CTkFrame(right, corner_radius=15)
info.pack(padx=30, pady=25, fill="x")

ctk.CTkLabel(
    info,
    text="Informations du modèle",
    font=("Arial", 18, "bold")
).pack(pady=(15, 5))

ctk.CTkLabel(
    info,
    text="Modèles : CNN\nBases : MNIST + EMNIST Letters\nClasses : 0 à 9 et A à Z\nPrécision chiffres : 99.17%\nPrécision lettres : 92.43%",
    font=("Arial", 16),
    justify="left"
).pack(pady=(5, 15))

btn_quit = ctk.CTkButton(
    right,
    text="Quitter",
    command=app.destroy,
    width=120,
    height=38,
    fg_color="#555555",
    hover_color="#333333"
)
btn_quit.pack(pady=5)

app.mainloop()