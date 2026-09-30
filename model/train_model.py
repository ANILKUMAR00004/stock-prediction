import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def train_lstm(df):

    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    sequence_length = 60

    if len(scaled_data) <= sequence_length:
        raise ValueError(
            "Not enough stock data to train the model. "
            "Please select a larger date range (at least 120 days)."
        )

    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError(
            "Not enough stock data to train the model. "
            "Please select a larger date range."
        )

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Keep track of dates matching the y targets
    target_dates = df.index[sequence_length:]

    # CHRONOLOGICAL SPLIT (Strict temporal order: first 80% train, last 20% test)
    # Eliminates data leakage and look-ahead bias
    split_idx = int(len(X) * 0.8)
    if split_idx == 0 or split_idx >= len(X):
        split_idx = max(1, len(X) - 1)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = target_dates[split_idx:]

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64))
    model.add(Dropout(0.2))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    # Early stopping to prevent overfitting and speed up training
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=32,
        epochs=15,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("model/stock_model.keras")

    return model, scaler, scaled_data, X_test, y_test, dates_test
