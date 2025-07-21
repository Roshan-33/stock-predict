import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📈 Stock Price Prediction App (Lightweight Version)")

# ✅ User Inputs
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

        # ✅ Prepare Data
        df['Prediction'] = df['Close'].shift(-1)
        X = np.array(df[['Close']])
        y = np.array(df['Prediction'])

        X = X[:-1]
        y = y[:-1]

        # ✅ Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # ✅ Train Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # ✅ Predictions
        predictions = model.predict(X_test)

        # ✅ Plot Results
        st.subheader("Actual vs Predicted Stock Price")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test, color="blue", label="Actual Price")
        ax.plot(predictions, color="red", label="Predicted Price")
        ax.set_title(f"{ticker} Stock Price Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
