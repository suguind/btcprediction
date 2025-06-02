import torch
from prometheus_client import Gauge
from datetime import datetime
import pytz  # Use pytz for timezone handling

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Global parameters
asset = "XXBTZUSD"  # Corrected to Kraken's pair format (XBT/USD is not the API format)
interval = 60  # Hourly data
start_time = datetime(2014, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
window_length = 24
fixed_horizons = [1, 6, 12, 24, 168]
num_quantiles = 3
quantiles = [0.05, 0.5, 0.95]
num_assets = 1
data_type = 'trades'  # Added parameter for data type

# Prometheus metrics
PREDICTION_LATENCY = Gauge('prediction_latency_seconds', 'Time taken for predictions')
PREDICTION_ERROR = Gauge('prediction_error', 'Prediction error')
MODEL_DRIFT = Gauge('model_drift', 'Metric for model drift')
FEATURE_DRIFT = Gauge('feature_drift', 'KL divergence of feature distributions')