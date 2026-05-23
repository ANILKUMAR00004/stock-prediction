import numpy as np
from model.train_model import train_lstm

def predict_stock(df):

    model, scaler, X_test, y_test = train_lstm(df)

    predictions = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    # ACCURACY
    accuracy = 100 - (
        np.mean(np.abs((y_test - predictions) / y_test)) * 100
    )

    future_days = 30

    last_sequence = X_test[-1]
    future_predictions = []

    current_sequence = last_sequence.copy()

    for _ in range(future_days):

        current_sequence_reshaped = np.reshape(
            current_sequence,
            (1, current_sequence.shape[0], 1)
        )

        next_prediction = model.predict(
            current_sequence_reshaped,
            verbose=0
        )

        future_predictions.append(next_prediction[0,0])
        current_sequence = np.delete(current_sequence, 0)

        current_sequence = np.append(
            current_sequence,
            next_prediction[0,0]
        )

    future_predictions = np.array(future_predictions)

    future_predictions = scaler.inverse_transform(
        future_predictions.reshape(-1,1)
    )

    return (
        y_test.flatten(),
        predictions.flatten(),
        future_predictions.flatten(),
        accuracy
    )