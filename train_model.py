import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/weather.csv")

# Remove missing values
data = data.dropna()

# Convert target column
data['RainTomorrow'] = data['RainTomorrow'].map({
    'Yes': 1,
    'No': 0
})

# Select features
features = [
    'MinTemp',
    'MaxTemp',
    'Humidity9am',
    'Pressure9am',
    'WindSpeed9am'
]

X = data[features]
y = data['RainTomorrow']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save model
pickle.dump(
    model,
    open("rain_model.pkl", "wb")
)

print("Model saved successfully!")