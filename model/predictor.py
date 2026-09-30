import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.train_model import train_lstm

def predict_stock(df):

    model, scaler, scaled_data, X_test, y_test, dates_test = train_lstm(df)

    predictions = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # METRICS
    # 1. MAPE & Accuracy
    mape = float(np.mean(np.abs((y_test - predictions) / y_test)) * 100)
    accuracy = float(max(0.0, 100.0 - mape))

    # 2. RMSE & MAE
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))

    # 3. Directional Accuracy (Did model correctly predict up/down day-over-day movement?)
    if len(y_test) > 1:
        actual_diff = y_test[1:] - y_test[:-1]
        predicted_diff = predictions[1:] - y_test[:-1]
        correct_directions = np.sum((actual_diff * predicted_diff) > 0)
        directional_accuracy = float((correct_directions / len(actual_diff)) * 100)
    else:
        directional_accuracy = 50.0

    # FUTURE FORECAST (30 days autoregressive from the true historical end)
    future_days = 30
    sequence_length = 60

    # Use the actual latest 60 days of scaled history
    current_sequence = scaled_data[-sequence_length:, 0].copy()
    future_predictions = []

    for _ in range(future_days):
        current_sequence_reshaped = np.reshape(
            current_sequence,
            (1, sequence_length, 1)
        )

        next_prediction = model.predict(
            current_sequence_reshaped,
            verbose=0
        )

        pred_val = next_prediction[0, 0]
        future_predictions.append(pred_val)

        # Shift sequence forward with newly predicted point
        current_sequence = np.append(current_sequence[1:], pred_val)

    future_predictions = np.array(future_predictions)
    future_predictions = scaler.inverse_transform(
        future_predictions.reshape(-1, 1)
    ).flatten()

    metrics = {
        "accuracy": accuracy,
        "mape": mape,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy
    }

    return (
        dates_test,
        y_test,
        predictions,
        future_predictions,
        metrics
    )
