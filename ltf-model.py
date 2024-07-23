import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Conv1D, GlobalMaxPooling1D, Bidirectional, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import ta
import logging

logging.basicConfig(level=logging.INFO)

# Load the data
data = pd.read_csv('bitcoin_prices_2024_07_23.csv')

# Generate TA data
#data['SMA'] = ta.trend.sma_indicator(data['binance_spot Price'], window=12)
#data['RSI'] = ta.momentum.rsi(data['binance_spot Price'], window=12)
#data['MACD'] = ta.trend.macd_diff(data['binance_spot Price'])
#data['Volatility'] = data['binance_spot Price'].pct_change().rolling(window=12).std()
    
# Select relevant features focusing on real-time data
features = [
    'binance_spot Price', 'binance_spot Highest Bid', 'binance_spot Lowest Ask', 'binance_spot Bid-Ask Spread', 
    'binance_spot Bid Volume', 'binance_spot Ask Volume', 'binance_spot Order Book Imbalance', 'binance_spot Trade Volume', 'binance_spot Trade Count',
    'binance_futures Price', 'binance_futures Highest Bid', 'binance_futures Lowest Ask', 'binance_futures Bid-Ask Spread', 
    'binance_futures Bid Volume', 'binance_futures Ask Volume', 'binance_futures Order Book Imbalance', 'binance_futures Trade Volume', 'binance_futures Trade Count',
    'coinbase Price', 'coinbase Highest Bid', 'coinbase Lowest Ask', 'coinbase Bid-Ask Spread', 'coinbase Bid Volume', 'coinbase Ask Volume', 
    'coinbase Order Book Imbalance', 'coinbase Trade Volume', 'coinbase Trade Count',
    'kraken Price', 'kraken Highest Bid', 'kraken Lowest Ask', 'kraken Bid-Ask Spread', 'kraken Bid Volume', 'kraken Ask Volume', 
    'kraken Order Book Imbalance', 'kraken Trade Volume', 'kraken Trade Count',
    'okx Price', 'okx Highest Bid', 'okx Lowest Ask', 'okx Bid-Ask Spread', 'okx Bid Volume', 'okx Ask Volume', 
    'okx Order Book Imbalance', 'okx Trade Volume', 'okx Trade Count',
    'bybit Price', 'bybit Highest Bid', 'bybit Lowest Ask', 'bybit Bid-Ask Spread', 'bybit Bid Volume', 'bybit Ask Volume', 
    'bybit Order Book Imbalance', 'bybit Trade Volume', 'bybit Trade Count'
]

target = 'binance_spot Price'

# Create a new column for price movement
data['Price_Movement'] = data[target].diff().fillna(0)

# Normalize the features and the new target variable
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + ['Price_Movement']])

# Create sequences of data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]
        y = data[i+seq_length, -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 3
x, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split the data into training and testing sets
split = int(0.9 * len(x))
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

# Build the model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQ_LENGTH, x.shape[2])),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    Dense(1)
])

# Compile the model with a learning rate scheduler
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, batch_size=32, epochs=1000, validation_split=0.2)

# Evaluate the model
model.evaluate(x_test, y_test)

# Save the model
model.save('bitcoin_ltf_model.keras')

# Predict
predictions = model.predict(x_test)
logging.info(f"x_test: {x_test.flatten()}")
logging.info(f"Final Predictions: {predictions.flatten()}")

# Scale
y_test_scaled = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], data_scaled.shape[1]-1)), y_test.reshape(-1, 1)), axis=1))[:, -1]
predictions_scaled = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], data_scaled.shape[1]-1)), predictions), axis=1))[:, -1]

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test_scaled, color='blue', label='Actual Price Movements')
plt.plot(predictions_scaled, color='red', label='Predicted Price Movements')
plt.xlabel('Time')
plt.ylabel('Price Movement')
plt.legend()
plt.show()
