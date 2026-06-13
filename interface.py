import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mnist_model.keras")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Reconnaissance de chiffres par IA")
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

def reconnaitre():
    img = image.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array, verbose=0)
    chiffre = int(np.argmax(prediction))
    confiance = float(np.max(prediction) * 100)

    img.save("dernier_test.png")

    if confiance < 70:
        label_resultat.configure(text="?")
        label_message.configure(text="Symbole non reconnu")
        label_confiance.configure(text=f"Confiance : {confiance:.2f}%")
    else:
        label_resultat.configure(text=str(chiffre))
        label_message.configure(text="Chiffre reconnu")
        label_confiance.configure(text=f"Confiance : {confiance:.2f}%")

def effacer():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_width, canvas_height], fill=0)
    label_resultat.configure(text="-")
    label_message.configure(text="Dessinez un chiffre de 0 à 9")
    label_confiance.configure(text="Confiance : -")

header = ctk.CTkFrame(app, height=90, corner_radius=0)
header.pack(fill="x")

title = ctk.CTkLabel(
    header,
    text="Reconnaissance automatique des chiffres manuscrits",
    font=("Arial", 28, "bold")
)
title.pack(pady=(18, 5))

subtitle = ctk.CTkLabel(
    header,
    text="Projet IA - Réseau de neurones CNN entraîné sur MNIST",
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
    text="Dessinez un chiffre de 0 à 9",
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
    text="Type : CNN\nBase : MNIST\nClasses : 0 à 9\nPrécision finale : 99.17%",
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