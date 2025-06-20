import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Configuration ---
SEQUENCE_LENGTH = 60 # Number of past days to consider for LSTM prediction
PREDICTION_DAYS = 7  # Number of future days to predict
LSTM_EPOCHS = 10     # Number of training epochs for LSTM (keep low for demo)
TEST_SIZE = 0.2      # Percentage of data for testing LSTM model

# --- Data Acquisition ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid excessive API calls
def get_historical_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.warning(f"No data found for ticker: {ticker}. Please check the ticker symbol.")
            return pd.DataFrame()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60) # Cache current price for 1 minute
def get_current_price(ticker):
    """
    Fetches the current closing price of a stock.
    """
    try:
        # For current price, fetch '1d' period and get the last 'Close' price
        ticker_info = yf.Ticker(ticker)
        data = ticker_info.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception:
        return None

# --- AI Prediction Model ---
def create_sequences(data, sequence_length):
    """
    Creates sequences for LSTM input.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

def train_lstm_model(ticker_data, sequence_length=SEQUENCE_LENGTH, epochs=LSTM_EPOCHS):
    """
    Trains an LSTM model for stock price prediction.
    """
    if ticker_data.empty:
        st.warning("No data to train the model.")
        return None, None

    # Use 'Close' prices for prediction
    data = ticker_data['Close'].values.reshape(-1, 1)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    if len(scaled_data) < sequence_length + 1:
        st.warning(f"Not enough historical data ({len(scaled_data)} days) to create sequences of length {sequence_length}. Need at least {sequence_length + 1} days.")
        return None, None

    X, y = create_sequences(scaled_data, sequence_length)

    # Reshape X for LSTM input [samples, timesteps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data into training and testing sets
    train_size = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output layer for predicting one value

    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner(f"Training AI model for {ticker_data.index[0].year}-{ticker_data.index[-1].year}... This might take a moment."):
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
    st.success("Model training complete!")

    # Evaluate the model (optional, for real-world you'd use a separate validation set)
    # If you want to show test loss:
    # loss = model.evaluate(X_test, y_test, verbose=0)
    # st.write(f"Model Test Loss: {loss:.4f}")

    return model, scaler

def predict_future_trend(model, scaler, last_sequence, num_predictions=PREDICTION_DAYS):
    """
    Predicts future stock prices using the trained LSTM model.
    """
    predicted_prices = []
    current_sequence = last_sequence.reshape(1, last_sequence.shape[0], 1)

    for _ in range(num_predictions):
        predicted_scaled_price = model.predict(current_sequence, verbose=0)[0, 0]
        # Inverse transform to get actual price
        predicted_price = scaler.inverse_transform([[predicted_scaled_price]])[0, 0]
        predicted_prices.append(predicted_price)

        # Update the sequence for the next prediction (autoregressive prediction)
        current_sequence = np.append(current_sequence[:, 1:, :], [[[predicted_scaled_price]]], axis=1)

    return predicted_prices

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="AI Stock Predictor")

# Custom CSS for a bit more styling
st.markdown("""
<style>
.main-header {
    font-size: 3em;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 30px;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.2rem;
    font-weight: bold;
}
.metric-value {
    font-size: 2.5em !important;
    color: #007BFF;
}
.metric-label {
    font-size: 1.2em !important;
    color: #555;
}

.stTabs {
    display: flex;
    justify-content: center;
    width: 100%;
}

.stTabs [data-baseweb="tab-list"] {
    max-width: 900px;
    width: 100%;
    justify-content: center;
    gap: 10px;
}

.main {
    padding: 20px;
    max-width: 1200px;
    margin: 20px auto;
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>AI Stock Predictor & Portfolio Manager</h1>", unsafe_allow_html=True)

# Initialize session state for portfolio and watchlist if not already present
if 'portfolio_holdings' not in st.session_state:
    st.session_state['portfolio_holdings'] = {} # {ticker: {'quantity': X, 'purchase_price': Y}}
if 'watchlist_stocks' not in st.session_state:
    st.session_state['watchlist_stocks'] = set()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Predict Trends", "My Portfolio", "Watchlist", "About & Info"])

with tab1:
    st.header("Predict Future Stock Trends")

    col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 1])
    with col_pred1:
        ticker_predict = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL)", "MSFT").upper()
    with col_pred2:
        prediction_days = st.slider("Future Days to Predict", 1, 14, PREDICTION_DAYS)
    with col_pred3:
        sequence_length = st.slider("LSTM Input History (Days)", 30, 90, SEQUENCE_LENGTH)
    
    # Calculate start date for data fetching (ensure enough data for training + sequence)
    # We need roughly (LSTM_EPOCHS * batch_size) + sequence_length days of data
    # A safe bet is a few years of data
    today = datetime.now()
    fetch_start_date = (today - timedelta(days=365 * 5)).strftime('%Y-%m-%d') # 5 years of data
    fetch_end_date = today.strftime('%Y-%m-%d')

    if st.button("Predict Trend for " + ticker_predict):
        if ticker_predict:
            stock_data = get_historical_data(ticker_predict, fetch_start_date, fetch_end_date)

            if not stock_data.empty:
                st.subheader(f"Historical Data for {ticker_predict}")
                st.write(stock_data.tail()) # Show last few rows of historical data

                model, scaler = train_lstm_model(stock_data, sequence_length=sequence_length)

                if model and scaler:
                    if len(stock_data) < sequence_length:
                        st.error(f"Not enough historical data ({len(stock_data)} days) to create the last sequence of length {sequence_length}. Please choose a smaller sequence length or a stock with more data.")
                    else:
                        last_sequence_data = stock_data['Close'].values[-sequence_length:].reshape(-1, 1)
                        scaled_last_sequence = scaler.transform(last_sequence_data)
                        
                        predicted_prices = predict_future_trend(model, scaler, scaled_last_sequence, num_predictions=prediction_days)

                        st.subheader(f"Predicted Future Prices for {ticker_predict} (Next {prediction_days} Trading Days):")
                        prediction_dates = [today + timedelta(days=i) for i in range(1, prediction_days + 1)]
                        prediction_df = pd.DataFrame({
                            'Date': prediction_dates,
                            'Predicted Close Price ($)': predicted_prices
                        })
                        prediction_df['Date'] = prediction_df['Date'].dt.date # Display only date
                        st.table(prediction_df)

                        # Plotting historical and predicted prices
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                                                mode='lines', name='Historical Close',
                                                line=dict(color='blue')))
                        
                        # Add predicted data
                        fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Close Price ($)'],
                                                mode='lines+markers', name='Predicted Close',
                                                line=dict(color='red', dash='dash'),
                                                marker=dict(symbol='circle', size=8)))

                        fig.update_layout(
                            title=f'{ticker_predict} Stock Price: Historical vs. Predicted',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified',
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not retrieve data for {ticker_predict}. Please check the ticker symbol and try again.")
        else:
            st.warning("Please enter a ticker symbol to predict.")

with tab2:
    st.header("My Portfolio")
    
    st.subheader("Add/Update Stock Holding")
    col_add1, col_add2, col_add3 = st.columns(3)
    with col_add1:
        add_ticker = st.text_input("Ticker Symbol", key="add_ticker_port", value="").upper()
    with col_add2:
        add_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="add_quantity_port")
    with col_add3:
        add_purchase_price = st.number_input("Purchase Price ($)", min_value=0.01, value=100.00, format="%.2f", key="add_price_port")

    if st.button("Add/Update Holding"):
        if add_ticker and add_quantity > 0 and add_purchase_price > 0:
            if add_ticker in st.session_state['portfolio_holdings']:
                # Update logic (weighted average purchase price)
                current_qty = st.session_state['portfolio_holdings'][add_ticker]['quantity']
                current_total_cost = st.session_state['portfolio_holdings'][add_ticker]['quantity'] * st.session_state['portfolio_holdings'][add_ticker]['purchase_price']
                
                new_total_qty = current_qty + add_quantity
                new_total_cost = current_total_cost + (add_quantity * add_purchase_price)
                
                st.session_state['portfolio_holdings'][add_ticker]['quantity'] = new_total_qty
                st.session_state['portfolio_holdings'][add_ticker]['purchase_price'] = new_total_cost / new_total_qty
                st.success(f"Updated {add_quantity} shares of {add_ticker}. New total: {new_total_qty} shares at avg ${st.session_state['portfolio_holdings'][add_ticker]['purchase_price']:.2f}.")
            else:
                st.session_state['portfolio_holdings'][add_ticker] = {'quantity': add_quantity, 'purchase_price': add_purchase_price}
                st.success(f"Added {add_quantity} shares of {add_ticker} at ${add_purchase_price:.2f} to portfolio.")
        else:
            st.warning("Please fill in all fields (Ticker, Quantity, Purchase Price) correctly.")

    st.subheader("Remove Stock from Portfolio")
    col_rem1, col_rem2 = st.columns(2)
    with col_rem1:
        remove_ticker = st.text_input("Ticker to Remove", key="remove_ticker_port", value="").upper()
    with col_rem2:
        remove_quantity = st.number_input("Quantity to Remove (0 for all)", min_value=0, value=0, step=1, key="remove_quantity_port")
    
    if st.button("Remove Holding"):
        if remove_ticker:
            if remove_ticker in st.session_state['portfolio_holdings']:
                current_held_qty = st.session_state['portfolio_holdings'][remove_ticker]['quantity']
                if remove_quantity == 0 or remove_quantity >= current_held_qty:
                    del st.session_state['portfolio_holdings'][remove_ticker]
                    st.success(f"Removed all shares of {remove_ticker} from portfolio.")
                elif remove_quantity < current_held_qty:
                    st.session_state['portfolio_holdings'][remove_ticker]['quantity'] -= remove_quantity
                    st.success(f"Removed {remove_quantity} shares of {remove_ticker}. Remaining: {st.session_state['portfolio_holdings'][remove_ticker]['quantity']} shares.")
                else:
                    st.warning("Quantity to remove exceeds current holding.")
            else:
                st.warning(f"{remove_ticker} not found in your portfolio.")
        else:
            st.warning("Please enter a ticker symbol to remove.")
    
    st.markdown("---")
    st.subheader("Current Portfolio Holdings & Value")
    if not st.session_state['portfolio_holdings']:
        st.info("Your portfolio is currently empty. Add some stocks above!")
    else:
        portfolio_data = []
        total_portfolio_value = 0.0
        total_profit_loss = 0.0

        tickers_to_fetch = list(st.session_state['portfolio_holdings'].keys())
        
        # Fetch current prices for all portfolio stocks in one go
        current_prices_df = yf.download(tickers_to_fetch, period="1d")['Close']

        for ticker, data in st.session_state['portfolio_holdings'].items():
            current_price = None
            if isinstance(current_prices_df, pd.Series): # Single ticker result
                if ticker in current_prices_df.index: # Handle case where yf.download returns a Series for single ticker
                    current_price = current_prices_df.loc[ticker]
                else:
                    current_price = current_prices_df.iloc[-1] if not current_prices_df.empty else None
            elif isinstance(current_prices_df, pd.DataFrame): # Multiple tickers result
                if not current_prices_df.empty and ticker in current_prices_df.columns:
                    current_price = current_prices_df[ticker].iloc[-1]
            
            if current_price is not None:
                current_holding_value = data['quantity'] * current_price
                profit_loss = current_holding_value - (data['quantity'] * data['purchase_price'])
                total_portfolio_value += current_holding_value
                total_profit_loss += profit_loss

                portfolio_data.append({
                    "Ticker": ticker,
                    "Quantity": data['quantity'],
                    "Purchase Price ($)": f"{data['purchase_price']:.2f}",
                    "Current Price ($)": f"{current_price:.2f}",
                    "Current Value ($)": f"{current_holding_value:.2f}",
                    "Profit/Loss ($)": f"{profit_loss:.2f}"
                })
            else:
                portfolio_data.append({
                    "Ticker": ticker,
                    "Quantity": data['quantity'],
                    "Purchase Price ($)": f"{data['purchase_price']:.2f}",
                    "Current Price ($)": "N/A",
                    "Current Value ($)": "N/A",
                    "Profit/Loss ($)": "N/A"
                })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True)

        col_total_val, col_total_pl = st.columns(2)
        with col_total_val:
            st.metric(label="Total Portfolio Value", value=f"${total_portfolio_value:,.2f}", delta=None) # Delta can be added with daily change
        with col_total_pl:
            st.metric(label="Total Profit/Loss", value=f"${total_profit_loss:,.2f}", delta=f"{total_profit_loss:,.2f}") # Streamlit handles delta color

with tab3:
    st.header("Watchlist")
    
    st.subheader("Add Stock to Watchlist")
    watchlist_add_ticker = st.text_input("Enter Stock Ticker to Watch", key="watchlist_add", value="").upper()
    if st.button("Add to Watchlist "):
        if watchlist_add_ticker:
            st.session_state['watchlist_stocks'].add(watchlist_add_ticker)
            st.success(f"{watchlist_add_ticker} added to watchlist.")
        else:
            st.warning("Please enter a ticker symbol.")

    st.subheader("Remove Stock from Watchlist")
    watchlist_remove_ticker = st.text_input("Enter Stock Ticker to Remove", key="watchlist_remove", value="").upper()
    if st.button("Remove from Watchlist "):
        if watchlist_remove_ticker in st.session_state['watchlist_stocks']:
            st.session_state['watchlist_stocks'].remove(watchlist_remove_ticker)
            st.success(f"{watchlist_remove_ticker} removed from watchlist.")
        elif watchlist_remove_ticker:
            st.warning(f"{watchlist_remove_ticker} not in your watchlist.")
        else:
            st.warning("Please enter a ticker symbol.")

    st.markdown("---")
    st.subheader("Current Watchlist")
    if not st.session_state['watchlist_stocks']:
        st.info("Your watchlist is empty. Add some stocks above!")
    else:
        watchlist_display_data = []
        tickers_to_fetch_watchlist = list(st.session_state['watchlist_stocks'])
        current_prices_watchlist_df = yf.download(tickers_to_fetch_watchlist, period="1d")['Close']

        for ticker in sorted(list(st.session_state['watchlist_stocks'])):
            current_price = None
            if isinstance(current_prices_watchlist_df, pd.Series): # Single ticker result
                if ticker in current_prices_watchlist_df.index:
                     current_price = current_prices_watchlist_df.loc[ticker]
                else:
                    current_price = current_prices_watchlist_df.iloc[-1] if not current_prices_watchlist_df.empty else None
            elif isinstance(current_prices_watchlist_df, pd.DataFrame): # Multiple tickers result
                if not current_prices_watchlist_df.empty and ticker in current_prices_watchlist_df.columns:
                    current_price = current_prices_watchlist_df[ticker].iloc[-1]

            if current_price is not None:
                watchlist_display_data.append({"Ticker": ticker, "Current Price ($)": f"{current_price:.2f}"})
            else:
                watchlist_display_data.append({"Ticker": ticker, "Current Price ($)": "N/A (Error or No Data)"})
        
        watchlist_df = pd.DataFrame(watchlist_display_data)
        st.dataframe(watchlist_df, use_container_width=True)

with tab4:
    st.header("About This AI Stock Predictor")
    st.markdown("""
    This application provides a basic framework for stock trend prediction, portfolio management, and watchlist tracking. It's built using Python with Streamlit for the user interface, `yfinance` for stock data, and `TensorFlow` (Keras) for the LSTM prediction model.
    """)

    st.subheader("How the Prediction Model Works (Simplified)")
    st.markdown(f"""
    The core of the stock prediction feature uses a **Long Short-Term Memory (LSTM)** neural network, a type of recurrent neural network well-suited for time-series forecasting.

    Here's a simplified breakdown of the process:
    1.  **Data Collection:** Historical 'Close' prices for the selected stock are fetched from Yahoo Finance. We typically use the last **{sequence_length} days** for each prediction sequence.
    2.  **Data Preprocessing:** The historical prices are scaled (normalized) between 0 and 1. This helps the neural network train more effectively.
    3.  **Sequence Creation:** The scaled data is then transformed into sequences. For instance, to predict day 61's price, the model is trained on sequences of the previous 60 days' prices.
    4.  **Model Training:** The LSTM neural network is trained on these sequences to learn the patterns and relationships within the historical data. The `epochs` setting (currently set to **{LSTM_EPOCHS}**) determines how many times the model iterates over the training data.
    5.  **Future Prediction:** After training, the model takes the most recent `{sequence_length}` days of actual stock prices, predicts the next day's price, then uses that prediction as part of the input to predict the day after, and so on for **{PREDICTION_DAYS} future days**. The scaled predictions are then inverse-transformed back to actual dollar values.

    **Important Considerations and Limitations:**
    * **Stock Market Volatility:** The stock market is influenced by countless unpredictable factors (economic news, geopolitical events, company announcements, investor sentiment, "black swan" events). No AI model can perfectly predict these.
    * **Model Simplification:** This is a relatively simple LSTM model. Real-world, highly accurate prediction models often involve:
        * More complex neural network architectures.
        * Incorporation of many more features (e.g., trading volume, technical indicators like RSI, MACD, Bollinger Bands, fundamental company data, news sentiment analysis).
        * Ensemble methods (combining multiple prediction models).
    * **Past Performance ≠ Future Results:** Models learn from historical data. While patterns can emerge, they do not guarantee future outcomes.
    * **Data Quality & Recency:** Predictions are only as good as the data they're trained on. Delays in data or inaccuracies can affect results.
    * **Not Financial Advice:** This tool is for educational and experimental purposes only. **Do not use it for real financial trading decisions without consulting a qualified financial advisor and conducting thorough independent research.** Investing in the stock market carries significant risks, including the potential loss of principal.

    Feel free to experiment with different ticker symbols, sequence lengths, and prediction days. The longer the historical data period used for training, the more patterns the model might learn, but excessively long periods might also include irrelevant past market conditions.
    """)

    st.subheader("Future Enhancements")
    st.markdown("""
    Potential features to add in a more advanced version:
    * **Technical Indicators:** Displaying and incorporating popular indicators (Moving Averages, RSI, MACD).
    * **News Sentiment Analysis:** Integrating APIs to fetch news and analyze sentiment for specific stocks.
    * **Fundamental Data:** Showing key financial ratios (P/E, EPS, Market Cap) for companies.
    * **Risk Analysis:** Calculating metrics like Beta, Volatility, Value at Risk (VaR).
    * **Backtesting:** Allowing users to test hypothetical trading strategies on historical data.
    * **Alerts:** Notifying users when predicted prices hit certain thresholds.
    * **User Accounts & Data Persistence:** For a multi-user system, implement user authentication and a database (e.g., SQLite, PostgreSQL) to store portfolio and watchlist data persistently.
    """)