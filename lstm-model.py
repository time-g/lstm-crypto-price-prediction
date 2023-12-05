#!/usr/bin/env python3

import os
from pprint import pprint

import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import (indicator_append, plot_metrics_val,
                   plot_price_test_prediction, plot_test_prediction,
                   rolling_window, save_loss, save_model)


def lstm_model(n_timesteps, n_features):
    """the model proposed by feiyang liu"""
    input_shape = (n_timesteps, n_features)
    model = Sequential()
    model.add(LSTM(33, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(79))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.00616353),
        loss="mse",
        metrics=[RootMeanSquaredError(), "mae", "mape"],
    )
    return model


def main():
    ticker = "ETH-USD"
    df = pd.read_csv(f"data/{ticker}.csv", index_col="Date", parse_dates=True)
    df = indicator_append(df)
    print(df.head())

    scaler = MinMaxScaler()
    scaler_close = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    scaler_close.fit(df["Close"].to_numpy()[:, None])

    window_size = 10
    target_col = df.columns.to_list().index("Close")
    x, y = rolling_window(df_scaled, window_size, target_col)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )

    # delete close-price the target from features
    # x_train = np.delete(x_train, target_col, axis=1)
    # x_test = np.delete(x_test, target_col, axis=1)
    n_timesteps = x_train.shape[1]
    n_features = x_train.shape[-1]
    epochs = 98
    model = lstm_model(n_timesteps, n_features)
    print(f"train model <{'-' * 80 }")
    history = model.fit(
        x_train, y_train, epochs=epochs, validation_data=(x_test, y_test)
    )

    print(f"prediction <{'-' * 80}")
    prediction = model.predict(x_test)

    test_mse = mean_squared_error(y_test, prediction)
    test_rmse = np.sqrt(test_mse)
    loss = {
        "train": {
            "MSE": history.history["loss"][-1],
            "RMSE": history.history["root_mean_squared_error"][-1],
            "MAE": history.history["mae"][-1],
            "MAPE": history.history["mape"][-1],
        },
        "test": {
            "MSE": test_mse,
            "RMSE": test_rmse,
            "MAE": mean_absolute_error(y_test, prediction),
            "MAPE": mean_absolute_percentage_error(y_test, prediction),
        },
    }

    model_name = save_model(model, __file__, ticker)
    saved_model = load_model(model_name)
    save_loss(loss, __file__, ticker)
    pprint(loss)
    model.summary()

    fname = os.path.basename(__file__)
    fname = os.path.splitext(fname)[0]

    plot_metrics_val(
        history,
        "root_mean_squared_error",
        "loss",
        "mae",
        __file__,
        ticker=ticker,
        title=ticker,
    )
    plot_model(
        model,
        to_file=f"report/{fname}-plot.png",
        show_shapes=True,
        show_layer_names=False,
        dpi=200,
    )
    plot_price_test_prediction(
        df["Close"],
        scaler_close.inverse_transform(y_test[:, None]),
        scaler_close.inverse_transform(prediction),
        df.index,
        window_size,
        # fname=f"report/{fname}-prediction.png",
        fname=__file__,
        ticker=ticker,
        title=ticker,
    )
    plot_test_prediction(
        scaler_close.inverse_transform(y_test[:, None]),
        scaler_close.inverse_transform(prediction),
        # fname=f"report/{fname}-prediction2.png",
        fname=__file__,
        ticker=ticker,
        title=ticker,
    )


if __name__ == "__main__":
    main()
