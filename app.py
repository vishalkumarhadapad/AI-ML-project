from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
import os
import csv
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# local modules
from career_details import career_info
from career_icons import career_icons

app = Flask(__name__)
app.secret_key = "replace_this_with_a_strong_random_key"

# Load model and encoder (ensure these files exist after training)
model = joblib.load("career_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Ensure folders/files exist
os.makedirs("history", exist_ok=True)
users_file = os.path.join("auth_users.csv")
if not os.path.exists(users_file):
    with open(users_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["username","password_hash"])

history_file = os.path.join("history","user_history.csv")
if not os.path.exists(history_file):
    with open(history_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["username","math","programming","creativity","communication","analytical","problemsolving","leadership","predicted_career","timestamp"])

# Helper: register user (stores hashed password)
def register_user(username, password):
    password_hash = generate_password_hash(password)
    with open(users_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([username, password_hash])

# Helper: authenticate
def authenticate_user(username, password):
    if not os.path.exists(users_file):
        return False
    with open(users_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == username:
                return check_password_hash(row["password_hash"], password)
    return False

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        if not username or not password:
            return render_template("register.html", error="All fields required.")
        # check if exists
        with open(users_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["username"] == username:
                    return render_template("register.html", error="Username already exists.")
        register_user(username, password)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username").strip()
        password = request.form.get("password").strip()
        if authenticate_user(username, password):
            session["username"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/", methods=["GET","POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    result = None
    top3 = None
    details = None
    icon = None
    error = None

    if request.method == "POST":
        try:
            # get inputs as strings — allow empty (treated below)
            vals = []
            keys = ["Math","Programming","Creativity","Communication","Analytical","ProblemSolving","Leadership"]
            for k in keys:
                v = request.form.get(k, "").strip()
                if v == "":
                    # if empty => default neutral 5
                    v = "5"
                # try convert
                vv = int(float(v))
                if vv < 1 or vv > 10:
                    error = "Skill values must be integers between 1 and 10 (empty treated as 5)."
                    return render_template("index.html", error=error)
                vals.append(vv)

            arr = np.array(vals).reshape(1, -1)

            # top-1
            pred_idx = model.predict(arr)[0]
            result = label_encoder.inverse_transform([pred_idx])[0]

            # top-3 with probabilities
            probs = model.predict_proba(arr)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [{"career": label_encoder.inverse_transform([i])[0],
                     "score": round(probs[i]*100,2)} for i in top3_idx]

            details = career_info.get(result, None)
            icon = career_icons.get(result, "")

            # save to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(history_file, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([session["username"]] + vals + [result, timestamp])

        except Exception as e:
            error = "Please enter valid numbers for skills."
            return render_template("index.html", error=error)

    return render_template("index.html", result=result, top3=top3, details=details, icon=icon, error=error, username=session.get("username"))

@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    data = []
    with open(history_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["username"] == session["username"]:
                data.append(row)
    return render_template("history.html", data=data)

if __name__ == "__main__":
    app.run(debug=True)