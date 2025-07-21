import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 Stock Price Prediction App (LSTM Model)")

# ✅ Ticker Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, RELIANCE.NS):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Predict"):
    st.info("Fetching stock data...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found for this ticker or date range.")
    else:
        st.success(f"Data fetched for {ticker}!")

        # ✅ Preprocess
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # ✅ Train/Test Split
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - 60:]

        # ✅ Create Dataset Function
        def create_dataset(dataset, time_step=60):
            X, y = [], []
            for i in range(time_step, len(dataset)):
                X.append(dataset[i - time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_dataset(train_data)
        X_test, y_test = create_dataset(test_data)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # ✅ Build LSTM Model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")

        # ✅ Train Model
        st.info("Training the LSTM model... Please wait ⏳")
        model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=0)
        st.success("Training completed!")

        # ✅ Prediction
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # ✅ Plot Results
        st.subheader("Actual vs Predicted Stock Price")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(real_prices, color="blue", label="Actual Price")
        ax.plot(predictions, color="red", label="Predicted Price")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
