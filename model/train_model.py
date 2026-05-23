
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def train_lstm(df):

    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    sequence_length = 60

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential()

    model.add(LSTM(units=128, return_sequences=True,
                   input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=64))
    model.add(Dropout(0.2))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        verbose=1
    )

    model.save("model/stock_model.keras")

    return model, scaler, X_test, y_test