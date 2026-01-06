import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load combined CSV with landmarks + labels
data = pd.read_csv("isl_all_data.csv")  # CSV created from extract_all_landmarks.py

# Features and labels
X = data.iloc[:, :-1].values  # all landmark columns
y = data.iloc[:, -1].values   # gesture labels

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a slightly larger model for more gestures
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# Save trained model
model.save("isl_model_all.h5")  # updated model name

# Save label encoder
with open("label_encoder_all.pkl", "wb") as f:
    pickle.dump(le, f)

print("Unified model and label encoder saved successfully.")
