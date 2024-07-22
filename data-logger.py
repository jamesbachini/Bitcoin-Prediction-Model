import ccxt.pro as ccxt
import pandas as pd
from datetime import datetime
import asyncio

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

# Function to fetch order book metrics
async def fetch_order_book_metrics(exchange, symbol):
    order_book = await exchange.fetch_order_book(symbol)
    highest_bid = order_book['bids'][0][0] if order_book['bids'] else None
    lowest_ask = order_book['asks'][0][0] if order_book['asks'] else None
    bid_ask_spread = lowest_ask - highest_bid if highest_bid and lowest_ask else None
    bid_volume = sum([bid[1] for bid in order_book['bids']])
    ask_volume = sum([ask[1] for ask in order_book['asks']])
    order_book_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if bid_volume + ask_volume > 0 else None
    return highest_bid, lowest_ask, bid_ask_spread, bid_volume, ask_volume, order_book_imbalance

# Function to fetch trade metrics
async def fetch_trade_metrics(exchange, symbol):
    trades = await exchange.fetch_trades(symbol)
    trade_volume = sum([trade['amount'] for trade in trades])
    trade_count = len(trades)
    return trade_volume, trade_count

# Function to fetch prices
async def fetch_price(exchange, symbol):
    ticker = await exchange.fetch_ticker(symbol)
    return ticker['last']

# Function to fetch and log prices asynchronously
async def fetch_and_log_prices():
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
            
            # Save to CSV
            df.to_csv('bitcoin_prices_22.csv', index=False)
            
            # Wait for 1 second
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error fetching prices: {e}")

if __name__ == "__main__":
    asyncio.run(fetch_and_log_prices())
