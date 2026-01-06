import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

SEQUENCE_PATH = "sequences"

X = []
y = []

for file in os.listdir(SEQUENCE_PATH):
    if file.endswith(".npy"):
        X.append(np.load(os.path.join(SEQUENCE_PATH, file)))
        label = file.split("_")[0]
        y.append(label)

X = np.array(X)

le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 126)),
    LSTM(64),
    Dense(64, activation="relu"),
    Dense(y.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=30, batch_size=16)
model.save("dynamic_lstm_model.h5")

with open("dynamic_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ LSTM model trained")
