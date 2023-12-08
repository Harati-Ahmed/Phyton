
import requests
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import joblib
import matplotlib.pyplot as plt

# Configuration...
API_ENDPOINT = 'https://api.binance.com/api/v3/klines'
INTERVAL = '1h'
LIMIT = 1000

# Technical indicator parameters
TECHNICAL_PARAMETERS = {
    'BTCUSDT': {
        'RSI_PERIOD': 14,
        'MA_PERIOD_SHORT': 20,
        'MA_PERIOD_LONG': 50,
        'BB_STD_DEV': 2,
        'EMA_PERIOD': 20,
        'MACD_FAST_PERIOD': 12,
        'MACD_SLOW_PERIOD': 26,
    },
    'ETHUSDT': {
        'RSI_PERIOD': 14,
        'MA_PERIOD_SHORT': 20,
        'MA_PERIOD_LONG': 50,
        'BB_STD_DEV': 2,
        'EMA_PERIOD': 20,
        'MACD_FAST_PERIOD': 12,
        'MACD_SLOW_PERIOD': 26,
    },
    # Add more symbols and their parameters as needed
}

# Risk management parameters
RISK_PER_TRADE = 0.02  # Risk 2% of account balance per trade
STOP_LOSS_MULTIPLIER = 1.5  # Stop-loss set at 1.5 times the ATR
# Portfolio management parameters...
INITIAL_BALANCE_PER_ASSET = 5000  # Initial balance distributed equally among assets

# Logging configuration...
LOG_FILE = 'trading_bot.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def fetch_binance_data(symbol, interval, limit):
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        response = requests.get(API_ENDPOINT, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set timestamp as the index
        df.set_index('timestamp', inplace=True)

        return df
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        raise

def clean_data(data):
    # Perform any necessary data cleaning steps
    # (e.g., handling missing values, outliers)

def calculate_rsi(data, period):
    close_prices = data['close'].astype(float).values
    rsi = talib.RSI(close_prices, timeperiod=period)
    return rsi

def calculate_moving_averages(data, short_period, long_period):
    data['ma_short'] = data['close'].rolling(window=short_period).mean()
    data['ma_long'] = data['close'].rolling(window=long_period).mean()

def calculate_bollinger_bands(data, period, num_std_dev):
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['close'].astype(float).values, timeperiod=period, nbdevup=num_std_dev, nbdevdn=num_std_dev)

def calculate_ema(data, period):
    return data['close'].ewm(span=period, adjust=False).mean()

def calculate_macd(data, fast_period, slow_period):
    macd, signal, _ = talib.MACD(data['close'].astype(float).values, fastperiod=fast_period, slowperiod=slow_period)
    return macd, signal

# Add your provided code for training machine learning model
def train_machine_learning_model(data):
    target = data['signal'].shift(-1)
    features = data[['rsi', 'ma_short', 'ma_long', 'upper_band', 'lower_band', 'ema', 'macd', 'signal_line']].shift(-1)

    features = features.dropna()
    target = target.dropna()

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    logging.info(f"Machine Learning Model Metrics: Accuracy={accuracy:.2f}, Precision={precision:.2f}, Recall={recall:.2f}, F1 Score={f1:.2f}")

    model_filename = 'trained_model.joblib'
    joblib.dump(model, model_filename)
    logging.info(f"Trained model saved to {model_filename}")

    return model

def apply_risk_management(data):
    atr = talib.ATR(data['high'].astype(float).values, data['low'].astype(float).values, data['close'].astype(float).values, timeperiod=14)
    data['atr'] = atr

    data['position_size'] = (RISK_PER_TRADE * data['balance']) / (STOP_LOSS_MULTIPLIER * atr)
    data['stop_loss'] = data['close'] - (STOP_LOSS_MULTIPLIER * atr)

def log_signals(data):
    for index, row in data.iterrows():
        if row['combined_signal'] == 1:
            logging.info(f"Buy: {row['close']} - Balance: {row['balance']:.2f} - Stop Loss: {row['stop_loss']:.2f}")
        elif row['combined_signal'] == -1:
            logging.info(f"Sell: {row['close']} - Balance: {row['balance']:.2f} - Stop Loss: {row['stop_loss']:.2f}")

def generate_features(data, rsi_period, ma_period_short, ma_period_long, bb_std_dev, ema_period, macd_fast_period, macd_slow_period):
    data['rsi'] = calculate_rsi(data, rsi_period)
    calculate_moving_averages(data, ma_period_short, ma_period_long)
    calculate_bollinger_bands(data, ma_period_long, bb_std_dev)
    data['ema'] = calculate_ema(data, ema_period)
    data['macd'], data['signal_line'] = calculate_macd(data, macd_fast_period, macd_slow_period)

def backtest(data):
    initial_balance = data['balance'][0]
    balance = initial_balance
    position = 0

    for index, row in data.iterrows():
        if row['combined_signal'] == 1 and position != 1:
            position = 1
            balance *= (row['close'] / row['close'].shift(1))

        elif row['combined_signal'] == -1 and position != -1:
            position = -1
            balance *= (row['close'] / row['close'].shift(1))

    final_balance = data['balance'].iloc[-1]
    returns = (final_balance - initial_balance) / initial_balance * 100

    logging.info(f"Backtest Results: Initial Balance={initial_balance:.2f}, Final Balance={final_balance:.2f}, Returns={returns:.2f}%")

def visualize_signals_and_performance(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['close'], label='Closing Price', linewidth=2)
    plt.scatter(data[data['combined_signal'] == 1].index, data[data['combined_signal'] == 1]['close'], label='Buy Signal', marker='^', color='g')
    plt.scatter(data[data['combined_signal'] == -1].index, data[data['combined_signal'] == -1]['close'], label='Sell Signal', marker='v', color='r')
    plt.title('Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['balance'], label='Account Balance', linewidth=2)
    plt.title('Account Balance Over Time')
    plt.xlabel('Date')
    plt.ylabel('Account Balance')
    plt.legend()
    plt.show()

def manage_portfolio(data):
    asset_balance = INITIAL_BALANCE_PER_ASSET
    asset_positions = {symbol: 0 for symbol in TECHNICAL_PARAMETERS.keys()}

    for index, row in data.iterrows():
        for symbol, params in TECHNICAL_PARAMETERS.items():
            combined_signal_column = f'{symbol.lower()}_combined_signal'
            if row[combined_signal_column] == 1 and asset_positions[symbol] != 1:
                asset_positions[symbol] = 1
                asset_balance *= (row[symbol] / row[symbol].shift(1))

            elif row[combined_signal_column] == -1 and asset_positions[symbol] != -1:
                asset_positions[symbol] = -1
                asset_balance *= (row[symbol] / row[symbol].shift(1))

    total_balance = sum(asset_balance for asset_balance in asset_positions.values())
    returns = (total_balance - len(TECHNICAL_PARAMETERS) * INITIAL_BALANCE_PER_ASSET) / (len(TECHNICAL_PARAMETERS) * INITIAL_BALANCE_PER_ASSET) * 100

    logging.info(f"Portfolio Management Results: Initial Balance={len(TECHNICAL_PARAMETERS) * INITIAL_BALANCE_PER_ASSET:.2f}, Final Balance={total_balance:.2f}, Returns={returns:.2f}%")

def main():
    for symbol, params in TECHNICAL_PARAMETERS.items():
        symbol_data = fetch_binance_data(symbol, INTERVAL, LIMIT)
        symbol_data['balance'] = INITIAL_BALANCE_PER_ASSET
        clean_data(symbol_data)
        generate_features(symbol_data, **params)
        apply_risk_management(symbol_data)

        machine_learning_model = train_machine_learning_model(symbol_data)

        symbol_data[f'{symbol.lower()}_ml_signal'] = machine_learning_model.predict(
            symbol_data[['rsi', 'ma_short', 'ma_long', 'upper_band', 'lower_band', 'ema', 'macd', 'signal_line']])
        symbol_data[f'{symbol.lower()}_combined_signal'] = symbol_data['signal'] + symbol_data[
            f'{symbol.lower()}_ml_signal']

        log_signals(symbol_data)
        visualize_signals_and_performance(symbol_data)

    # Combine all symbols into a portfolio dataframe
    portfolio_data = pd.concat([symbol_data[[f'{symbol.lower()}_combined_signal', symbol.lower()]] for symbol in TECHNICAL_PARAMETERS.keys()], axis=1)
    manage_portfolio(portfolio_data)

if __name__ == "__main__":
    main()
