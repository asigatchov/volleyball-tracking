import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


sequence_length, num_features = 10, 6  # Примерные значения, замените на свои
model = Sequential(
    [
        LSTM(64, input_shape=(sequence_length, num_features)),  # Например, (10, 6)
        Dense(32, activation="relu"),
        Dense(4, activation="softmax"),  # 4 класса: подача, прием, передача, атака
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
