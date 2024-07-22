import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Set timestamp as index
    df.set_index('Timestamp', inplace=True)
    
    # Select features for prediction (you can modify this based on your analysis)
    features = ['binance_spot Price', 'binance_futures Price', 'coinbase Price', 'kraken Price', 'bitfinex Price', 'okx Price', 'bybit Price']
    
    # Create a new dataframe with selected features
    data = df[features]
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    return normalized_data, scaler

# 2. Create sequences for LSTM input
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :])
        y.append(data[i + sequence_length, 0])  # Predicting binance_spot Price
    return np.array(X), np.array(y)

# 3. Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 4. Train the model
def train_model(X_train, y_train, X_val, y_val):
    model = build_model((X_train.shape[1], X_train.shape[2]))
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history

# 5. Make predictions
def make_predictions(model, X_test, scaler):
    predictions = model.predict(X_test)
    # Inverse transform the predictions
    predictions = scaler.inverse_transform(np.hstack([predictions, np.zeros((len(predictions), 6))]))[:, 0]
    return predictions

# 6. Visualize results
def plot_results(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.legend()
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    data, scaler = load_and_preprocess_data('bitcoin_prices.csv')
    
    # Create sequences
    sequence_length = 24  # You can adjust this value
    X, y = create_sequences(data, sequence_length)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train the model
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = make_predictions(model, X_test, scaler)
    
    # Inverse transform the actual values
    actual = scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((len(y_test), 6))]))[:, 0]
    
    # Plot results
    plot_results(actual, predictions)

    print("Mean Absolute Error:", np.mean(np.abs(actual - predictions)))
    print("Root Mean Squared Error:", np.sqrt(np.mean((actual - predictions)**2)))