import threading
from queue import Queue
import pandas as pd
import krakenex
import time
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import RobustScaler
import json
import os
import numpy as np
import pandas_ta as pta
import logging
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from .config import asset, interval, start_time, data_type

# Configure logging for detailed debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define fixed horizons for forecasting
fixed_horizons = [1, 6, 12, 24, 168]  # Hours: 1h, 6h, 12h, 24h, 1 week

# Thread-safe DataStore class
class DataStore:
    def __init__(self, maxsize=100):
        self.lock = threading.Lock()
        self.raw = None
        self.processed = None
        self.feature_cols = None

    def update(self, raw, processed, feature_cols=None):
        with self.lock:
            self.raw = raw.copy() if raw is not None else None
            self.processed = processed.copy() if processed is not None else None
            self.feature_cols = feature_cols.copy() if feature_cols is not None else None

    def get(self):
        with self.lock:
            return (
                self.raw.copy() if self.raw is not None else None,
                self.processed.copy() if self.processed is not None else None,
                self.feature_cols.copy() if self.feature_cols is not None else None
            )

data_store = DataStore(maxsize=100)
data_queue = Queue(maxsize=100)

def fetch_kraken_trades_data(pair, start_time, mode='train'):
    """
    Fetch trade data from Kraken API or load from Parquet file.
    - 'train' mode: Use existing file if available, otherwise fetch and save.
    - 'predict' mode: Fetch last year's data.
    """
    parquet_file = "kraken_xxbtzusd_trades.parquet"
    
    if mode == 'train' and os.path.exists(parquet_file):
        try:
            df = pd.read_parquet(parquet_file)
            logging.info(f"Loaded {len(df)} trades from {parquet_file}")
            logging.info(f"Raw price stats: min={df['price'].min()}, max={df['price'].max()}, std={df['price'].std()}, unique={df['price'].nunique()}")
            logging.info(f"First 5 prices: {df['price'].head(5).to_list()}")
            logging.info(f"Raw data index type: {type(df.index)}")
            logging.info(f"Raw data time range: {df.index.min()} to {df.index.max()}")
            logging.info(f"Number of NaN prices: {df['price'].isnull().sum()}")
            logging.info(f"First 5 timestamps: {df.index[:5].tolist()}")
            logging.info(f"Last 5 timestamps: {df.index[-5:].tolist()}")
            return df
        except Exception as e:
            logging.error(f"Error loading Parquet: {e}. Fetching new data.")
    
    elif mode == 'predict':
        start_time = datetime.now(timezone.utc) - timedelta(days=365)
        logging.info(f"Fetching prediction data from {start_time}")

    try:
        api = krakenex.API()
        since = str(int(start_time.timestamp() * 1e9))  # Nanoseconds
        all_trades = []
        seen_trade_ids = set()

        while True:
            resp = api.query_public('Trades', {'pair': pair, 'since': since})
            if 'error' in resp and resp['error']:
                raise Exception(f"API error: {resp['error']}")
            pair_data_key = [k for k in resp['result'].keys() if k != 'last'][0]
            trades = resp['result'][pair_data_key]
            logging.info(f"Received {len(trades)} trades")
            if not trades:
                break
            new_trades = [t for t in trades if t[6] not in seen_trade_ids]
            if not new_trades:
                break
            all_trades.extend(new_trades)
            seen_trade_ids.update(t[6] for t in new_trades)
            last_time = pd.to_datetime(new_trades[-1][2], unit='s', utc=True)
            logging.info(f"Fetched {len(new_trades)} trades, last: {last_time}")
            since = resp['result']['last']
            time.sleep(1)

        if not all_trades:
            raise ValueError("No trade data fetched")

        df = pd.DataFrame(all_trades, columns=['price', 'volume', 'time', 'buy_sell', 'market_limit', 'misc', 'trade_id'])
        logging.info(f"First 5 raw 'time' values: {df['time'].head(5).to_list()}")
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        df = df[['price', 'volume']].astype(float).sort_index()
        df = df[~df.index.duplicated(keep='first')]
        logging.info(f"Fetched {len(df)} trades")
        logging.info(f"Raw price stats: min={df['price'].min()}, max={df['price'].max()}, std={df['price'].std()}, unique={df['price'].nunique()}")
        logging.info(f"First 5 prices: {df['price'].head(5).to_list()}")
        logging.info(f"Raw data index type: {type(df.index)}")
        logging.info(f"Raw data time range: {df.index.min()} to {df.index.max()}")
        logging.info(f"Number of NaN prices: {df['price'].isnull().sum()}")
        logging.info(f"First 5 timestamps: {df.index[:5].tolist()}")
        logging.info(f"Last 5 timestamps: {df.index[-5:].tolist()}")

        if mode == 'train':
            pq.write_table(pa.Table.from_pandas(df), parquet_file)
            logging.info(f"Saved to {parquet_file}")

        return df

    except Exception as e:
        logging.error(f"Fetch error: {e}")
        raise

def validate_ohlc_data(df):
    """
    Validate OHLC data for NaN or infinite values.
    """
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing OHLC columns")
    
    if df[required_cols].isnull().any().any() or np.isinf(df[required_cols]).any().any():
        logging.error(f"OHLC data contains NaN or infinite values: {df[required_cols].isnull().sum()}")
        raise ValueError("OHLC data contains NaN or infinite values")
    
    logging.info("OHLC data validated successfully")
    logging.info(f"Close price stats after filling: min={df['close'].min()}, max={df['close'].max()}, std={df['close'].std()}, unique={df['close'].nunique()}")

def shared_preprocessing(data, robust_scaling=True, scaler=None):
    """
    Preprocess data with OHLC aggregation, technical indicators, and scaling.
    Includes validation and detailed diagnostics.
    """
    try:
        data = data.copy()
        if data.empty or 'price' not in data.columns or 'volume' not in data.columns:
            raise ValueError("Invalid input data")
        logging.info(f"Input shape: {data.shape}")

        # Validate raw data index
        if not isinstance(data.index, pd.DatetimeIndex):
            logging.error(f"Data index is not a DatetimeIndex: {type(data.index)}")
            raise ValueError("Data index must be a DatetimeIndex")
        if not data.index.is_monotonic_increasing:
            logging.error("Raw data index is not monotonic increasing")
            raise ValueError("Raw data index must be monotonic increasing")
        if data['price'].isnull().any():
            logging.error(f"Raw data contains {data['price'].isnull().sum()} NaN prices")
            raise ValueError("Raw data contains NaN prices")

        # Align time index with 4-hour bins
        start_time = data.index.min().floor('4h')  # e.g., 2014-01-01 00:00
        end_time = data.index.max().ceil('4h')    # e.g., 2025-04-22 00:00
        time_index = pd.date_range(start=start_time, end=end_time, freq='4h', tz='UTC')

        # Aggregate to 4-hourly OHLC
        ohlc_data = data['price'].resample('4h').ohlc()
        volume_data = data['volume'].resample('4h').sum().fillna(0)

        # Reindex to ensure full time coverage and fill NaNs
        ohlc_data = ohlc_data.reindex(time_index).ffill().bfill()
        volume_data = volume_data.reindex(time_index).fillna(0)

        data = ohlc_data.copy()
        data['volume'] = volume_data
        logging.info(f"OHLC shape: {data.shape}")

        # Validate OHLC data
        validate_ohlc_data(data)

        # Time-based features
        data['hour'] = data.index.hour
        data['minute'] = data.index.minute
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # Past-based percentage changes with epsilon to avoid division by zero
        epsilon = 1e-10
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[f'{col}_pct'] = (data[col] - data[col].shift(1)) / (data[col].shift(1) + epsilon)
            data[f'{col}_pct'] = data[f'{col}_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
        logging.info(f"Shape after pct changes: {data.shape}")
        logging.info(f"close_pct stats: min={data['close_pct'].min()}, max={data['close_pct'].max()}, mean={data['close_pct'].mean()}, std={data['close_pct'].std()}, zeros={len(data[data['close_pct'] == 0])}")

        # Technical indicators with minimal periods
        min_periods = 1
        data['price_ewma_9'] = pta.ema(data['close'], length=9, min_periods=min_periods)
        data['price_ewma_21'] = pta.ema(data['close'], length=21, min_periods=min_periods)
        data['volume_ewma_9'] = pta.ema(data['volume'], length=9, min_periods=min_periods)
        data['volume_ewma_21'] = pta.ema(data['volume'], length=21, min_periods=min_periods)
        data['rsi'] = pta.rsi(data['close'], length=14, min_periods=min_periods)
        macd = pta.macd(data['close'], fast=12, slow=26, signal=9, min_periods=min_periods)
        data['macd'] = macd.iloc[:, 0]
        data['signal'] = macd.iloc[:, 1]
        data['histogram'] = macd.iloc[:, 2]
        adx = pta.adx(high=data['high'], low=data['low'], close=data['close'], length=14, min_periods=min_periods)
        data['adx_14'] = adx.iloc[:, 2]
        data['cci'] = pta.cci(high=data['high'], low=data['low'], close=data['close'], length=20, min_periods=min_periods)
        data['atr'] = pta.atr(high=data['high'], low=data['low'], close=data['close'], length=14, min_periods=min_periods)
        data['obv'] = pta.obv(data['close'], data['volume'])
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

        # Fill NaNs in technical indicators
        technical_cols = [
            'price_ewma_9', 'price_ewma_21', 'volume_ewma_9', 'volume_ewma_21',
            'rsi', 'macd', 'signal', 'histogram', 'adx_14', 'cci', 'atr', 'obv', 'vwap'
        ]
        data[technical_cols] = data[technical_cols].fillna(0)
        logging.info(f"Shape after indicators: {data.shape}")

        full_features = [
            'open_pct', 'high_pct', 'low_pct', 'close_pct', 'volume_pct',
            'price_ewma_9', 'price_ewma_21', 'volume_ewma_9', 'volume_ewma_21',
            'rsi', 'macd', 'signal', 'histogram', 'adx_14', 'cci', 'atr', 'obv', 'vwap',
            'close'
        ]

        # Pre-scaling checks for NaN or infinite values
        if data[full_features].isnull().values.any():
            logging.error(f"NaN values in features before scaling: {data[full_features].isnull().sum()}")
            raise ValueError("NaN values in features")
        if np.isinf(data[full_features]).values.any():
            inf_counts = np.isinf(data[full_features]).sum()
            logging.error(f"Infinite values in features before scaling: {inf_counts}")
            data[full_features] = data[full_features].replace([np.inf, -np.inf], np.nan).fillna(0)

        if scaler is None:
            scaler = RobustScaler()
            scaler.fit(data[full_features])
        data[full_features] = np.clip(scaler.transform(data[full_features]), -10, 10)

        if np.any(np.isnan(data[full_features])) or np.any(np.isinf(data[full_features])):
            raise ValueError("Invalid values after scaling")

        return data, full_features, scaler

    except Exception as e:
        logging.error(f"Preprocessing error: {e}")
        raise

def validate_features(df: pd.DataFrame, feature_list: list, scaler: RobustScaler = None) -> pd.DataFrame:
    missing = [feat for feat in feature_list if feat in df.columns]
    if missing:
        logging.warning(f"Missing features filled with 0: {missing}")
        for feat in missing:
            df[feat] = 0.0
    df = df[feature_list]
    if scaler and hasattr(scaler, "n_features_in_") and df.shape[1] != scaler.n_features_in_:
        raise ValueError(f"Feature count mismatch: {df.shape[1]} vs {scaler.n_features_in_}")
    return df

def persist_features(feature_list: list, path="data_persistence/features_used.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_list, f)

def preprocessing_worker():
    counter = 0
    while True:
        try:
            new_row = data_queue.get(timeout=5)
            current_raw, _, _ = data_store.get()
            if current_raw is None:
                current_raw = new_row
            else:
                current_raw = pd.concat([current_raw, new_row])
            processed, features, scaler = shared_preprocessing(current_raw.copy())
            data_store.update(current_raw, processed, features)
            counter += 1
            if counter % 100 == 0:
                persist_features(features)
            del new_row
            gc.collect()
        except Exception as e:
            logging.error(f"Worker error: {e}")
            time.sleep(1)

def sample_sliding_windows(data, window_length, n_samples):
    try:
        global fixed_horizons  # Use global fixed_horizons
        max_horizon = max(fixed_horizons)
        # Convert hours to number of 4-hour intervals
        max_horizon_rows = int(max_horizon / 4)
        if max_horizon % 4 != 0:
            logging.warning(f"Maximum horizon {max_horizon} hours not divisible by 4-hour interval; rounding down")
        
        data = data.copy()
        feature_cols = data.columns.tolist()
        if 'close' not in feature_cols:
            logging.error("'close' column is missing from the data")
            return []

        # Calculate volatility and drop NaN values
        data['volatility'] = data['close_pct'].rolling(window=window_length).std()
        data = data.dropna()
        logging.info(f"Data shape after volatility calc: {data.shape}")

        # Log volatility statistics
        vol_stats = {
            'min': data['volatility'].min(),
            'max': data['volatility'].max(),
            'mean': data['volatility'].mean(),
            'unique': data['volatility'].nunique()
        }
        logging.info(f"Volatility stats: {vol_stats}")

        windows = []
        bin_choices = []

        if vol_stats['unique'] > 1:
            try:
                data['vol_bin'] = pd.qcut(data['volatility'], q=5, labels=False, duplicates='drop')
                bin_choices = data['vol_bin'].unique()
                logging.info(f"Quantile binning successful with {len(bin_choices)} bins")
            except ValueError:
                logging.warning("Quantile binning failed; trying equal-width bins")
                try:
                    data['vol_bin'] = pd.cut(data['volatility'], bins=5, labels=False, duplicates='drop')
                    bin_choices = data['vol_bin'].unique()
                    logging.info(f"Equal-width binning successful with {len(bin_choices)} bins")
                except Exception as e:
                    logging.warning(f"Equal-width binning failed: {e}; using random sampling")
        else:
            logging.info("Insufficient volatility variation; using random sampling")

        for _ in range(n_samples):
            if len(bin_choices) > 0:
                bin_choice = np.random.choice(bin_choices)
                bin_data = data[data['vol_bin'] == bin_choice]
            else:
                bin_data = data

            # Check if bin has enough data for window + horizon
            if len(bin_data) < window_length + max_horizon_rows:
                logging.debug(f"Bin {bin_choice if len(bin_choices) > 0 else 'all'} too small: {len(bin_data)} rows")
                continue

            # Generate a window
            idx = np.random.randint(window_length, len(bin_data) - max_horizon_rows)
            window = bin_data.iloc[idx - window_length:idx]

            # Validate the window
            if window.shape[0] == window_length and not window.isnull().values.any() and not np.isinf(window).values.any():
                windows.append(window)
            else:
                logging.debug(f"Invalid window: size={window.shape[0]}, NaN present={window.isnull().values.any()}, Inf present={np.isinf(window).values.any()}")

        # Fallback to random sampling if needed
        if len(windows) < n_samples:
            logging.warning(f"Generated {len(windows)} valid windows; filling with random sampling")
            while len(windows) < n_samples and len(data) >= window_length + max_horizon_rows:
                idx = np.random.randint(window_length, len(data) - max_horizon_rows)
                window = data.iloc[idx - window_length:idx]
                if window.shape[0] == window_length and not window.isnull().values.any() and not np.isinf(window).values.any():
                    windows.append(window)
                else:
                    logging.debug(f"Invalid random window: size={window.shape[0]}, NaN present={window.isnull().values.any()}, Inf present={np.isinf(window).values.any()}")

        logging.info(f"Total valid windows generated: {len(windows)}")
        if len(windows) == 0:
            logging.error("No valid windows generated")
        return windows

    except Exception as e:
        logging.error(f"Error in sample_sliding_windows: {e}")
        return []