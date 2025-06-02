import websocket
import json
import threading
import logging
import time
import pandas as pd
from .data_utils import data_queue
from .config import asset

def on_message(ws, message):
    try:
        msg = json.loads(message)
        if isinstance(msg, list) and len(msg) == 4 and msg[2] == "ohlc-60":
            _, ohlc_data, _, _ = msg
            time, _, open_, high, low, close, _, volume, _ = ohlc_data
            new_data = {
                "open_time": pd.to_datetime(float(time), unit='s', utc=True),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume)
            }
            new_row = pd.DataFrame([new_data])
            new_row.set_index("open_time", inplace=True)
            try:
                data_queue.put_nowait(new_row)
            except:
                logging.warning("Data queue full. Dropping data.")
    except Exception as e:
        logging.error(f"WebSocket on_message error: {e}")

def on_error(ws, error):
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logging.info("WebSocket closed. Reconnecting in 5 seconds...")
    time.sleep(5)
    start_websocket()

def on_open(ws):
    try:
        subscription = {
            "event": "subscribe",
            "pair": [asset],
            "subscription": {
                "name": "ohlc",
                "interval": 60
            }
        }
        ws.send(json.dumps(subscription))
    except Exception as e:
        logging.error(f"Error in on_open: {e}")

def start_websocket():
    try:
        ws = websocket.WebSocketApp(
            "wss://ws.kraken.com",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever()
    except Exception as e:
        logging.error(f"Error starting WebSocket: {e}")