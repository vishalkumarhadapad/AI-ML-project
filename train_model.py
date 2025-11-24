import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("dataset.csv")

# 7 Skill Columns
features = [
    "Math",
    "Programming",
    "Creativity",
    "Communication",
    "Analytical",
    "ProblemSolving",
    "Leadership"
]

# Feature Matrix (X) and Label (y)
X = df[features]
y = df["Career"]

# Encode Career Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

# Save Model + Encoder
joblib.dump(model, "career_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("MODEL TRAINED SUCCESSFULLY WITH ALL 7 FEATURES!")