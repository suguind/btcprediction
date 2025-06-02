import os
import time
import threading
import logging
import json
import subprocess
import socket
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner, RunConfig, TuneConfig
import torch
from prometheus_client import start_http_server
from retry import retry
import mlflow
from .data_utils import fetch_kraken_trades_data, shared_preprocessing, data_store, preprocessing_worker
from .train import train_supervised_model, build_model_from_config
from .predict import run_hourly_prediction
from .websocket import start_websocket
from .utils import seed_everything
from .config import device, asset, start_time, fixed_horizons, num_quantiles

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Enable MPS fallback
if torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Start Prometheus server
@retry(tries=3, delay=2)
def start_prometheus():
    try:
        start_http_server(8000)
        logging.info("Prometheus server started on port 8000.")
    except OSError as e:
        if e.errno == 48:
            logging.warning("Port 8000 in use; skipping Prometheus server.")
        else:
            raise

start_prometheus()

# Seed for reproducibility
seed_everything(42)

# Function to check if a port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return False
        except OSError:
            return True

# Start MLflow server programmatically
def start_mlflow_server():
    port = 5001
    tracking_uri = f"http://localhost:{port}"
    backend_store_uri = "sqlite:///mlflow.db"

    if is_port_in_use(port):
        logging.info(f"Port {port} is already in use; assuming MLflow server is running.")
        return None, tracking_uri

    logging.info(f"Starting MLflow server on port {port}...")
    process = subprocess.Popen(
        [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--backend-store-uri", backend_store_uri
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait for the server to start (up to 10 seconds)
    for _ in range(20):
        if not is_port_in_use(port):
            logging.info(f"MLflow server started successfully on {tracking_uri}")
            return process, tracking_uri
        time.sleep(0.5)

    stdout, stderr = process.communicate(timeout=5)
    logging.error(f"Failed to start MLflow server: {stdout} {stderr}")
    raise RuntimeError("MLflow server failed to start")

# Initialize Ray and MLflow
try:
    ray.init(ignore_reinit_error=True, num_cpus=1)  # Limit resources for stability
    logging.info("Ray initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Ray: {e}")
    raise

# Start MLflow server and set tracking URI
mlflow_process, tracking_uri = start_mlflow_server()
mlflow.set_tracking_uri(tracking_uri)
try:
    mlflow.set_experiment("crypto_forecasting")
    logging.info(f"Experiment set successfully with tracking URI: {tracking_uri}")
except mlflow.exceptions.MlflowException as e:
    logging.warning(f"Failed to set experiment with {tracking_uri}: {e}")
    logging.info("Falling back to local file-based tracking.")
    mlflow.set_tracking_uri("file:///Users/sugumardeva/mlruns")
    mlflow.set_experiment("crypto_forecasting")
    logging.info("Experiment set with local tracking.")

with mlflow.start_run():
    # Fetch initial training data
    @retry(tries=3, delay=5)
    def fetch_initial_data():
        try:
            data = fetch_kraken_trades_data(asset, start_time, mode='train')
            if data.empty or data.isnull().values.any():
                raise ValueError("Fetched data is empty or contains NaNs.")
            return data
        except Exception as e:
            logging.error(f"Data fetch failed: {e}")
            raise

    try:
        df_raw = fetch_initial_data()
        df_processed, feature_cols, global_scaler = shared_preprocessing(df_raw.copy(), robust_scaling=True)
        # Store training data in Ray's object store
        train_data_ref = ray.put(df_processed)
        feature_cols_ref = ray.put(feature_cols)
        logging.info("Initial training data fetched and processed successfully.")
    except Exception as e:
        logging.error(f"Failed to fetch or process initial training data: {e}")
        raise

    # Define Ray Tune search space, including data references
    search_space = {
        "feature_dim": len(feature_cols),  # Dynamically set based on feature_cols
        "dynamic_graph": tune.choice([True, False]),
        "gnn_layer_type": tune.choice(["SAGE", "GCN", "GAT"]),
        "use_lstm": tune.choice([True, False]),
        "tcn_channels": tune.choice([16, 32, 64]),
        "tcn_kernel_size": tune.choice([2, 3, 5]),
        "tcn_layers": tune.choice([2, 3, 4]),
        "gnn_out": tune.choice([64, 128]),
        "transformer_hidden": tune.choice([128, 256]),
        "transformer_layers": tune.qrandint(16, 48, 8),
        "n_heads": tune.choice([8, 16]),
        "dropout": tune.uniform(0.05, 0.15),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        "lr": tune.loguniform(5e-5, 2e-4),
        "meta_epochs": tune.choice([150, 200]),
        "ss_epochs": tune.choice([50, 100]),
        "n_tasks": tune.choice([100, 200]),
        "mask_ratio": tune.uniform(0.3, 0.4),
        "data_aug_noise": 0.01,
        "sft_epochs": 50,
        "train_data_ref": train_data_ref,
        "feature_cols_ref": feature_cols_ref
    }

    # Train or load model
    trained_model_path = os.getenv("MODEL_PATH", "trained_model.pth")
    best_config_path = os.getenv("CONFIG_PATH", "best_config.json")

    if os.path.exists(trained_model_path) and os.path.exists(best_config_path):
        logging.info("Loading existing model.")
        try:
            best_state = torch.load(trained_model_path, map_location=device)
            with open(best_config_path, "r") as f:
                best_config = json.load(f)
            mlflow.log_artifact(trained_model_path)
            mlflow.log_artifact(best_config_path)
            logging.info("Model and config loaded and logged to MLflow.")
        except Exception as e:
            logging.error(f"Failed to load model or config: {e}")
            raise
    else:
        logging.info("Starting Ray Tune training.")
        asha_scheduler = ASHAScheduler(metric="loss", mode="min", max_t=200, grace_period=10, reduction_factor=2)
        tuner = Tuner(
            train_supervised_model,
            param_space=search_space,
            run_config=RunConfig(
                name="train_supervised_model",
                log_to_file=("tune.log", "tune_error.log"),
                storage_path=os.path.abspath("./ray_results")
            ),
            tune_config=TuneConfig(
                num_samples=50,
                max_concurrent_trials=1,
                scheduler=asha_scheduler
            )
        )
        try:
            analysis = tuner.fit()
            best_result = analysis.get_best_trial(metric="loss", mode="min")
            best_config = best_result.config
            # Remove Ray object references from saved config
            config_to_save = {k: v for k, v in best_config.items() if k not in ["train_data_ref", "feature_cols_ref"]}
            with open(best_config_path, "w") as f:
                json.dump(config_to_save, f)
            if best_result.checkpoint:
                with best_result.checkpoint.as_directory() as checkpoint_dir:
                    cp_path = os.path.join(checkpoint_dir, "model.pth")
                    best_state = torch.load(cp_path, map_location=device)
                    torch.save(best_state, trained_model_path)
                    mlflow.log_artifact(trained_model_path)
                    mlflow.log_artifact(best_config_path)
                    logging.info("Model trained and artifacts logged to MLflow.")
            else:
                best_state = None
                logging.warning("No checkpoint found.")
        except Exception as e:
            logging.error(f"Ray Tune training failed: {e}")
            raise

    # Load best model
    try:
        model = build_model_from_config(config_to_save, fixed_horizons, num_quantiles)
        if best_state:
            model.load_state_dict(best_state)
            logging.info("Model loaded successfully.")
        else:
            logging.warning("No checkpoint; using untrained model.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

    # Export model for production
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, "scripted_model.pt")
        logging.info("Model scripted and saved.")
        mlflow.log_artifact("scripted_model.pt")
    except Exception as e:
        logging.warning(f"Scripting failed: {e}. Attempting tracing instead.")
        try:
            example_input = torch.randn(1, 24, 1, config_to_save["feature_dim"]).to(device)
            traced_model = torch.jit.trace(model, example_input)
            traced_model.save("traced_model.pt")
            logging.info("Model traced and saved.")
            mlflow.log_artifact("traced_model.pt")
        except Exception as trace_e:
            logging.error(f"Tracing failed: {trace_e}")

    # Fetch initial prediction data after training
    @retry(tries=3, delay=5)
    def fetch_prediction_data():
        try:
            pred_data = fetch_kraken_trades_data(asset, None, mode='predict')
            if pred_data.empty or pred_data.isnull().values.any():
                raise ValueError("Prediction data is empty or contains NaNs.")
            pred_processed, _, _ = shared_preprocessing(pred_data.copy(), robust_scaling=True, scaler=global_scaler)
            data_store.update(pred_data, pred_processed)
            logging.info("Initial prediction data fetched and processed successfully.")
            return pred_data, pred_processed
        except Exception as e:
            logging.error(f"Failed to fetch or process prediction data: {e}")
            raise

    try:
        pred_raw, pred_processed = fetch_prediction_data()
    except Exception as e:
        logging.error(f"Prediction data setup failed: {e}")
        raise

    # Thread execution for real-time operations
    def run_with_error_handling(target, name):
        while True:
            try:
                target()
            except Exception as e:
                logging.error(f"Error in {name}: {e}")
                time.sleep(5)

    ws_thread = threading.Thread(
        target=lambda: run_with_error_handling(start_websocket, "WebSocket"), daemon=True
    )
    preproc_thread = threading.Thread(
        target=lambda: run_with_error_handling(preprocessing_worker, "Preprocessing"), daemon=True
    )
    hourly_thread = threading.Thread(
        target=lambda: run_with_error_handling(run_hourly_prediction, "Hourly Prediction"), daemon=True
    )

    ws_thread.start()
    preproc_thread.start()
    hourly_thread.start()

    ws_thread.join()
    preproc_thread.join()
    hourly_thread.join()

# Cleanup
ray.shutdown()
logging.info("Ray shutdown completed.")
if mlflow_process:
    mlflow_process.terminate()
    logging.info("MLflow server terminated.")