from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Classical classifier wrapper
class ClassicalClassifier:
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return classification_report(y, preds)


# LSTM model builder (multi-class friendly)
def build_lstm_model(vocab_size: int,
                     embedding_dim: int = 128,
                     seq_len: int = 100,
                     num_classes: int = 1):
    """
    Build a simple LSTM model using TensorFlow Keras.
    If num_classes == 1 -> binary classification with sigmoid.
    If num_classes > 1 -> multi-class classification with softmax.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    except Exception as e:
        raise ImportError("TensorFlow is required to build the LSTM model") from e

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_len))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    if num_classes == 1:
        model.add(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
