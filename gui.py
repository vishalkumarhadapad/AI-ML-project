import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load model
model = joblib.load("career_model.pkl")
le = joblib.load("label_encoder.pkl")

# Predict career
def predict_career():
    try:
        values = [
            int(math_entry.get()),
            int(program_entry.get()),
            int(creativity_entry.get()),
            int(comm_entry.get()),
            int(analytical_entry.get())
        ]

        values = np.array(values).reshape(1, -1)
        pred = model.predict(values)
        career = le.inverse_transform(pred)[0]

        messagebox.showinfo("Career Recommendation", f"Best Career for You:\n\n{career}")

    except:
        messagebox.showerror("Error", "Please enter numbers between 1 to 10.")

# MAIN WINDOW
root = tk.Tk()
root.title("AI Career Recommendation System")
root.geometry("400x500")
root.configure(bg="#1e1e2e")

title_label = tk.Label(root, text="AI Career Recommender",
                       font=("Arial", 20, "bold"),
                       fg="white", bg="#1e1e2e")
title_label.pack(pady=20)

frame = tk.Frame(root, bg="#2e2e3e", bd=5, relief="ridge")
frame.pack(pady=10, padx=20)

def create_input(label_text):
    label = tk.Label(frame, text=label_text, font=("Arial", 12),
                     fg="white", bg="#2e2e3e")
    label.pack(pady=5)
    entry = tk.Entry(frame, font=("Arial", 12))
    entry.pack(pady=5)
    return entry

math_entry = create_input("Math (1 - 10)")
program_entry = create_input("Programming (1 - 10)")
creativity_entry = create_input("Creativity (1 - 10)")
comm_entry = create_input("Communication (1 - 10)")
analytical_entry = create_input("Analytical (1 - 10)")

predict_btn = tk.Button(root, text="Predict Career", command=predict_career,
                        font=("Arial", 14, "bold"),
                        bg="#4a90e2", fg="white", width=15)
predict_btn.pack(pady=20)

root.mainloop()