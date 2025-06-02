import random
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from sklearn.neighbors import NearestNeighbors
import logging
from .config import device

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def advanced_time_warp(x, num_control_points=4, sigma=0.2):
    T, D = x.shape
    control_points = np.linspace(0, T - 1, num=num_control_points)
    offsets = np.zeros(num_control_points)
    offsets[1:-1] = np.random.normal(0, sigma * T, size=num_control_points - 2)
    warped_control = control_points + offsets
    cs = CubicSpline(control_points, warped_control)
    new_indices = cs(np.arange(T))
    new_indices = np.clip(new_indices, 0, T - 1)
    warped = np.zeros_like(x)
    for d in range(D):
        warped[:, d] = np.interp(np.arange(T), new_indices, x[:, d])
    return warped

def run_backtest(df):
    try:
        close_prices = df["close_orig"].values
        peak = -np.inf
        max_dd = 0.0
        for price in close_prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
    except Exception as e:
        logging.error(f"Error in run_backtest: {e}")
        return 0.0

def compute_dynamic_edge_index(x, k=5):
    try:
        norm = x.norm(dim=1, keepdim=True) + 1e-8
        x_norm = x / norm
        sim_matrix = x_norm @ x_norm.t()
        edge_list = []
        T = x.size(0)
        for i in range(T):
            if i == 0:
                continue
            sim_row = sim_matrix[i, :i]
            topk = min(k, i)
            _, top_indices = torch.topk(sim_row, topk)
            for j in top_indices:
                edge_list.append([int(j), i])
                edge_list.append([i, int(j)])
        if edge_list:
            return torch.tensor(edge_list, dtype=torch.long, device=x.device).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)
    except Exception as e:
        logging.error(f"Error in compute_dynamic_edge_index: {e}")
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

def compute_dynamic_edge_index_efficient(x, k=5):
    try:
        x_np = x.detach().cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_np)
        _, indices = nbrs.kneighbors(x_np)
        edge_list = []
        T = x.size(0)
        for i in range(T):
            for j in indices[i]:
                if i != j:
                    edge_list.append([j, i])
                    edge_list.append([i, j])
        if edge_list:
            return torch.tensor(edge_list, dtype=torch.long, device=x.device).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)
    except Exception as e:
        logging.error(f"Error in compute_dynamic_edge_index_efficient: {e}")
        return torch.empty((2, 0), dtype=torch.long, device=x.device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def predict_with_uncertainty(x, model, n_samples=100):
    price_preds = []
    aux_preds = []
    model.train()
    with torch.no_grad():
        for _ in range(n_samples):
            pred_price, pred_aux, _ = model(x)
            price_preds.append(pred_price.cpu().numpy())
            aux_preds.append(pred_aux.cpu().numpy())
    return np.mean(price_preds, axis=0), np.std(price_preds, axis=0), \
           np.mean(aux_preds, axis=0), np.std(aux_preds, axis=0)

def calculate_dynamic_position_size(risk_per_trade: float, uncertainty, epsilon=1e-6) -> float:
    from .data_utils import data_store
    _, processed = data_store.get()
    if processed is not None and "atr" in processed.columns:
        current_atr = processed["atr"].iloc[-1]
        return min(risk_per_trade / (current_atr * (uncertainty + epsilon)), 0.1)
    else:
        return 0.0

def analyze_feature_importance(df, feature_cols):
    correlations = {}
    for col in feature_cols:
        correlations[col] = df[col].corr(df['close_pct'])
    sorted_corr = sorted(correlations.items(), key=lambda item: abs(item[1]), reverse=True)
    logging.info("Feature importance (Pearson correlation with close_pct):")
    for col, corr in sorted_corr:
        logging.info(f"{col}: {corr:.4f}")
    return correlations

def log_prediction_results(actual_price, predicted_price):
    error = abs(actual_price - predicted_price)
    logging.info(f"Actual: {actual_price}, Predicted: {predicted_price}, Error: {error}")
    return error