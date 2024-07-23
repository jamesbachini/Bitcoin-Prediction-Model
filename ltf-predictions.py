import ccxt.pro as ccxt
import pandas as pd
from datetime import datetime
import asyncio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize exchanges
exchanges = {
    'binance_spot': ccxt.binance(),
    'binance_futures': ccxt.binance({'options': {'defaultType': 'future'}}),
    'coinbase': ccxt.coinbase(),
    'kraken': ccxt.kraken(),
    'okx': ccxt.okx(),
    'bybit': ccxt.bybit(),
}

symbols = {
    'binance_spot': 'BTC/USDT',
    'binance_futures': 'BTC/USDT',
    'coinbase': 'BTC/USD',
    'kraken': 'BTC/USD',
    'okx': 'BTC/USDT',
    'bybit': 'BTC/USDT',
}

# Load the trained model
model = load_model('bitcoin_ltf_model.keras')

# Load the scaler used during model training
scaler = MinMaxScaler()
data_sample = pd.read_csv('bitcoin_prices_2024_07_23.csv')  # Load sample data to fit the scaler
scaler.fit(data_sample.iloc[:, 1:])

# Function to fetch order book metrics
async def fetch_order_book_metrics(exchange, symbol):
    try:
        order_book = await exchange.fetch_order_book(symbol)
        highest_bid = order_book['bids'][0][0] if order_book['bids'] else None
        lowest_ask = order_book['asks'][0][0] if order_book['asks'] else None
        bid_ask_spread = lowest_ask - highest_bid if highest_bid and lowest_ask else None
        bid_volume = sum([bid[1] for bid in order_book['bids']])
        ask_volume = sum([ask[1] for ask in order_book['asks']])
        order_book_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if bid_volume + ask_volume > 0 else None
        return highest_bid, lowest_ask, bid_ask_spread, bid_volume, ask_volume, order_book_imbalance
    except Exception as e:
        logging.error(f"Error fetching order book metrics for {symbol}: {e}")
        return None, None, None, None, None, None

# Function to fetch trade metrics
async def fetch_trade_metrics(exchange, symbol):
    try:
        trades = await exchange.fetch_trades(symbol)
        trade_volume = sum([trade['amount'] for trade in trades])
        trade_count = len(trades)
        return trade_volume, trade_count
    except Exception as e:
        logging.error(f"Error fetching trade metrics for {symbol}: {e}")
        return None, None

# Function to fetch prices
async def fetch_price(exchange, symbol):
    try:
        ticker = await exchange.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {e}")
        return None

# Function to preprocess data for prediction
def preprocess_data_for_prediction(data):
    try:
        data_scaled = scaler.transform(data)
        seq_length = 3  # Match the sequence length used in model training
        x_input = []
        for i in range(len(data_scaled) - seq_length + 1):
            x = data_scaled[i:i+seq_length, :]
            x_input.append(x)
        return np.array(x_input)
    except Exception as e:
        logging.error(f"Error preprocessing data for prediction: {e}")
        return np.array([])

predict_array = []

# Function to fetch, preprocess and predict prices asynchronously
async def fetch_preprocess_predict():
    data = []
    
    while True:
        try:
            tasks = []
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row = [now]

            for exchange_name, exchange in exchanges.items():
                symbol = symbols[exchange_name]
                tasks.append(fetch_price(exchange, symbol))
                tasks.append(fetch_order_book_metrics(exchange, symbol))
                tasks.append(fetch_trade_metrics(exchange, symbol))
            
            results = await asyncio.gather(*tasks)
            
            for i in range(0, len(results), 3):
                row.append(results[i])  # price
                row.extend(results[i+1])  # order book metrics
                row.extend(results[i+2])  # trade metrics
                
            data.append(row)
            
            # Define column names
            columns = ['Timestamp']
            for exchange_name in exchanges.keys():
                columns.extend([
                    f'{exchange_name} Price', 
                    f'{exchange_name} Highest Bid', f'{exchange_name} Lowest Ask', 
                    f'{exchange_name} Bid-Ask Spread', f'{exchange_name} Bid Volume', 
                    f'{exchange_name} Ask Volume', f'{exchange_name} Order Book Imbalance',
                    f'{exchange_name} Trade Volume', f'{exchange_name} Trade Count'
                ])
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=columns)
            #logging.info(f"Data collected: {df.tail(3)}")  # Log the last 3 rows of data
            
            # Preprocess the data for prediction
            if len(df) >= 3:
                recent_data = df.iloc[-3:, 1:]  # Use the last 3 rows for prediction
                x_input = preprocess_data_for_prediction(recent_data)
                
                if x_input.size > 0:
                    # Predict the price movement
                    predictions = model.predict(x_input)
                    predicted_movement = predictions[-1][0]
                    current_price = row[1]  # Assuming the price is the second element in the row
                    predicted_price = current_price + predicted_movement

                    predict_array.append(predicted_movement)  # Add the new value
                    if len(predict_array) > 12:  # Check if the list exceeds 12 items
                        predict_array.pop(0)  # Remove the first item
                    mean_value = np.mean(predict_array)  # Calculate the mean of the list
                    prediction_minus_mean = predicted_movement - mean_value

                    logging.info(f"Current Price:    ${current_price:.2f}")
                    logging.info(f"Predicted Price:  ${predicted_price:.2f}")
                    logging.info(f"Predicted Change: ${predicted_movement:.2f}")
                    logging.info(f"Mean Value:       ${mean_value:.2f}")
                    logging.info(f"Adjusted:         ${prediction_minus_mean:.2f}")
                    logging.info(f"Direction:        {'UP' if prediction_minus_mean > 0 else 'DOWN' if prediction_minus_mean < 0 else 'FLAT'}")
                    logging.info(f"Trade:            {'LONG' if prediction_minus_mean > 0.05 else 'SHORT' if prediction_minus_mean < -0.05 else 'CLOSE'}")

                else:
                    logging.warning("x_input is empty after preprocessing.")
            
            # Wait for 1 second
            await asyncio.sleep(1)
            
        except Exception as e:
            logging.error(f"Error in fetch_preprocess_predict: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_preprocess_predict())
