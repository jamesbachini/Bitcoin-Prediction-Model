import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('bitcoin_prices.csv')

# Select relevant features
features = [
    'binance_spot Price', 'binance_spot Highest Bid', 'binance_spot Lowest Ask', 'binance_spot Bid-Ask Spread', 
    'binance_spot Bid Volume', 'binance_spot Ask Volume', 'binance_spot Order Book Imbalance', 'binance_spot Trade Volume', 'binance_spot Trade Count',
    'binance_futures Price', 'binance_futures Highest Bid', 'binance_futures Lowest Ask', 'binance_futures Bid-Ask Spread', 
    'binance_futures Bid Volume', 'binance_futures Ask Volume', 'binance_futures Order Book Imbalance', 'binance_futures Trade Volume', 'binance_futures Trade Count',
    'coinbase Price', 'coinbase Highest Bid', 'coinbase Lowest Ask', 'coinbase Bid-Ask Spread', 'coinbase Bid Volume', 'coinbase Ask Volume', 
    'coinbase Order Book Imbalance', 'coinbase Trade Volume', 'coinbase Trade Count',
    'kraken Price', 'kraken Highest Bid', 'kraken Lowest Ask', 'kraken Bid-Ask Spread', 'kraken Bid Volume', 'kraken Ask Volume', 
    'kraken Order Book Imbalance', 'kraken Trade Volume', 'kraken Trade Count',
    'bitfinex Price', 'bitfinex Highest Bid', 'bitfinex Lowest Ask', 'bitfinex Bid-Ask Spread', 'bitfinex Bid Volume', 'bitfinex Ask Volume', 
    'bitfinex Order Book Imbalance', 'bitfinex Trade Volume', 'bitfinex Trade Count',
    'okx Price', 'okx Highest Bid', 'okx Lowest Ask', 'okx Bid-Ask Spread', 'okx Bid Volume', 'okx Ask Volume', 
    'okx Order Book Imbalance', 'okx Trade Volume', 'okx Trade Count',
    'bybit Price', 'bybit Highest Bid', 'bybit Lowest Ask', 'bybit Bid-Ask Spread', 'bybit Bid Volume', 'bybit Ask Volume', 
    'bybit Order Book Imbalance', 'bybit Trade Volume', 'bybit Trade Count'
]

target = 'binance_spot Price'

# Normalize the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + [target]])

# Create sequences of data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]
        y = data[i+seq_length, -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 60  # Using the past 60 timesteps to predict the next price
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

# Evaluate the model
model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_test)

# Inverse transform the predictions
y_test_scaled = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], data_scaled.shape[1]-1)), y_test.reshape(-1, 1)), axis=1))[:, -1]
predictions_scaled = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], data_scaled.shape[1]-1)), predictions), axis=1))[:, -1]

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test_scaled, color='blue', label='Actual Prices')
plt.plot(predictions_scaled, color='red', label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
