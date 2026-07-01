import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("sample_dataset.csv")

# Rename columns
df.columns = ["Age", "Income", "CreditScore", "Gender", "Target"]

# Encode Gender
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Features & Target
X = df.drop("Target", axis=1)
y = df["Target"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save model
joblib.dump(model, "sample_model.pkl")

print("Model Saved Successfully!")