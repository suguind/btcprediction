import time
import numpy as np
import torch
import logging
from datetime import datetime, timedelta, timezone
from .model import HybridTCNGNNTransformerModel
from .utils import predict_with_uncertainty
from .data_utils import data_store
from .config import device, PREDICTION_LATENCY, FEATURE_DRIFT, fixed_horizons, num_quantiles, window_length

def predict_next_hour():
    try:
        _, proc = data_store.get()
        if proc is None or proc.shape[0] < window_length:
            logging.error("Not enough data for prediction.")
            return
        logging.info(f"Using data range for prediction: {proc.index.min()} to {proc.index.max()}")
        start_time = time.time()
        sample_window = proc.iloc[-window_length:][proc.columns[5:]].values  # Adjust based on actual feature columns
        current_price = proc["close_orig"].iloc[-1]
        sample_window = np.expand_dims(sample_window, axis=0)
        sample_window = np.expand_dims(sample_window, axis=2)
        x_sample = torch.tensor(sample_window, dtype=torch.float32).to(device)

        model = HybridTCNGNNTransformerModel(
            feature_dim=sample_window.shape[-1],  # Dynamically set based on input
            tcn_channels=32, tcn_kernel_size=3, tcn_layers=3,
            gnn_out=128, transformer_hidden=256, transformer_layers=16,
            n_heads=8, num_horizons=len(fixed_horizons), num_quantiles=num_quantiles,
            dropout=0.1
        ).to(device)
        # Note: In practice, load the trained model here instead of reinitializing

        model.eval()
        with torch.no_grad():
            price_quantiles, _, _ = model(x_sample)
            price_quantiles = price_quantiles.cpu().numpy()[0]

        elapsed = time.time() - start_time
        PREDICTION_LATENCY.set(elapsed)

        recent_window = proc.iloc[-window_length:][proc.columns[5:]].values
        feature_means = proc[proc.columns[5:]].mean().values
        recent_means = np.mean(recent_window, axis=0)
        drift = np.mean(np.abs(feature_means - recent_means))
        if drift > 0.1:
            logging.warning(f"Concept drift detected: {drift:.4f}")
        FEATURE_DRIFT.set(drift)

        output_lines = []
        output_lines.append(f"{'Horizon':>6} | {'Lower (5%)':>12} | {'Median (50%)':>12} | {'Upper (95%)':>12}")
        output_lines.append("-" * 50)
        for i, h in enumerate(fixed_horizons):
            lower = price_quantiles[i, 0]
            median = price_quantiles[i, 1]
            upper = price_quantiles[i, 2]
            output_lines.append(f"{h:6d}h | {lower:12.4%} | {median:12.4%} | {upper:12.4%}")
        logging.info("\n" + "\n".join(output_lines))

    except Exception as e:
        logging.error(f"Error in predict_next_hour: {e}")

def run_hourly_prediction():
    while True:
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        sleep_seconds = (next_hour - now).total_seconds()
        logging.info(f"Sleeping for {sleep_seconds:.2f} seconds until next hour...")
        time.sleep(sleep_seconds)
        predict_next_hour()