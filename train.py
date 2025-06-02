import ray
from ray import tune
import torch
import torch.optim as optim
from lion_pytorch import Lion
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch.nn as nn
import logging
import higher  # For MAML meta-learning
import torch.nn.functional as F
from .model import HybridTCNGNNTransformerModel, pinball_loss
from .utils import seed_everything, advanced_time_warp, run_backtest, init_weights
from .data_utils import sample_sliding_windows
from .config import device, fixed_horizons, num_quantiles, quantiles

class DeepThinkingController:
    def __init__(self, initial_config):
        self.config = initial_config
        self.metrics_history = {
            'train_loss': [], 'val_loss': [], 'grad_norm': [], 'uncertainty': [],
            'train_val_ratio': [], 'learning_rates': []
        }
        self.curriculum_stage = 0
        self.plateau_count = 0
        self.best_loss = float('inf')
        self.arch_changed = False
        self.online_adaptation_rate = self.config.get("online_adaptation_rate", 0.001)
        self.curriculum_horizons = [1, 3, 6, 12, 24]  # Progressive horizons for curriculum learning

    def update_config(self, current_metrics, backtest_drawdown=None, val_X=None, val_y=None, device=None):
        for k, v in current_metrics.items():
            if k in self.metrics_history:
                self.metrics_history[k].append(v)

        if current_metrics.get('grad_norm', 0) > 10:
            self.config['lr'] *= 0.9
            logging.info(f"Controller: High grad norm; reducing LR to {self.config['lr']}")

        if len(self.metrics_history['val_loss']) >= 5:
            recent_loss = self.metrics_history['val_loss'][-5:]
            if all(abs(l - recent_loss[0]) < 0.001 for l in recent_loss):
                self.plateau_count += 1
                if self.plateau_count >= 3:
                    new_lr = self.config['lr'] * 0.5
                    logging.info(f"Controller: Plateau detected; reducing LR to {new_lr}")
                    self.config['lr'] = new_lr
                    self.plateau_count = 0
            else:
                self.plateau_count = 0

        if len(self.metrics_history['val_loss']) >= 10:
            recent_val_loss = np.mean(self.metrics_history['val_loss'][-10:])
            if recent_val_loss < self.best_loss * 0.95 and self.curriculum_stage < len(self.curriculum_horizons) - 1:
                self.curriculum_stage += 1
                self.best_loss = recent_val_loss
                logging.info(f"Controller: Advancing to curriculum stage {self.curriculum_stage}, horizon {self.curriculum_horizons[self.curriculum_stage]}")

        if len(self.metrics_history['train_loss']) >= 10 and len(self.metrics_history['val_loss']) >= 10:
            ratio = np.mean(self.metrics_history['train_loss'][-10:]) / np.mean(self.metrics_history['val_loss'][-10:])
            self.metrics_history['train_val_ratio'].append(ratio)
            if ratio < 0.8:
                self.config['data_aug_noise'] *= 1.2
                logging.info(f"Controller: Increasing noise to {self.config['data_aug_noise']} (overfitting)")
            elif ratio > 1.2:
                self.config['data_aug_noise'] *= 0.8
                logging.info(f"Controller: Decreasing noise to {self.config['data_aug_noise']} (underfitting)")

        if backtest_drawdown is not None and backtest_drawdown > 0.1:
            new_transformer_layers = max(16, self.config["transformer_layers"] - 8)
            if new_transformer_layers != self.config["transformer_layers"]:
                logging.info(f"Controller: Reducing layers to {new_transformer_layers}")
                self.config["transformer_layers"] = new_transformer_layers
                self.arch_changed = True

        return self.config

    def online_adaptation(self, model, new_data, optimizer):
        model.train()
        optimizer.zero_grad()
        if not torch.is_tensor(new_data):
            new_data = torch.tensor(new_data, dtype=torch.float32).to(device)
        if new_data.dim() != 4:
            new_data = new_data.unsqueeze(0)
        pred, _, _ = model(new_data)
        target = torch.zeros_like(pred)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        for g in optimizer.param_groups:
            g['lr'] = self.online_adaptation_rate
        optimizer.step()
        logging.info(f"Online adaptation loss = {loss.item():.6f}")
        return loss.item()

def build_model_from_config(current_config, horizons, num_quantiles):
    supported_gnn_types = ["SAGE", "GCN", "GAT", "GraphConv", "ChebConv", "TransformerConv"]
    gnn_layer_type = current_config.get("gnn_layer_type", "SAGE")
    if gnn_layer_type not in supported_gnn_types:
        logging.error(f"Unsupported gnn_layer_type: {gnn_layer_type}")
        raise ValueError(f"Unsupported gnn_layer_type: {gnn_layer_type}")
    return HybridTCNGNNTransformerModel(
        feature_dim=current_config["feature_dim"],
        tcn_channels=current_config["tcn_channels"],
        tcn_kernel_size=current_config["tcn_kernel_size"],
        tcn_layers=current_config["tcn_layers"],
        gnn_out=current_config["gnn_out"],
        transformer_hidden=current_config["transformer_hidden"],
        transformer_layers=current_config["transformer_layers"],
        n_heads=current_config["n_heads"],
        num_horizons=len(horizons),
        num_quantiles=num_quantiles,
        dropout=current_config["dropout"],
        gnn_layer_type=gnn_layer_type,
        dynamic_graph=current_config.get("dynamic_graph", False),
        use_lstm=current_config.get("use_lstm", False),
        use_efficient_edge=current_config.get("use_efficient_edge", True)
    ).to(device)

def train_supervised_model(config):
    try:
        # Retrieve training data from Ray's object store
        df_processed = ray.get(config["train_data_ref"])
        feature_cols = ray.get(config["feature_cols_ref"])

        controller = DeepThinkingController(initial_config=config)
        logging.info(f"Data range: {df_processed.index.min()} to {df_processed.index.max()}")

        windows = sample_sliding_windows(df_processed, config.get("window_length", 24), n_samples=500)
        horizons = fixed_horizons
        all_windows, all_targets, all_aux_targets = [], [], []
        for window in windows:
            if window.shape[0] != config.get("window_length", 24):
                logging.debug(f"Skipping window with shape {window.shape}")
                continue
            last_index = window.index[-1]
            close_current = df_processed.loc[last_index, "close"]
            volume_current = df_processed.loc[last_index, "volume"]
            if not (np.isfinite(close_current) and np.isfinite(volume_current) and close_current > 0 and volume_current > 0):
                continue
            t_prices, t_volumes = [], []
            valid = True
            for h in horizons:
                future_time = last_index + pd.Timedelta(hours=h)
                if future_time not in df_processed.index:
                    valid = False
                    break
                close_future = df_processed.loc[future_time, "close"]
                volume_future = df_processed.loc[future_time, "volume"]
                if not (np.isfinite(close_future) and np.isfinite(volume_future)):
                    valid = False
                    break
                price_change = (close_future / close_current) - 1
                volume_change = (volume_future / volume_current) - 1
                if not (np.isfinite(price_change) and np.isfinite(volume_change)):
                    valid = False
                    break
                t_prices.append(price_change)
                t_volumes.append(volume_change)
            if valid and len(t_prices) == len(horizons):
                warped = advanced_time_warp(window[feature_cols].values, num_control_points=4, sigma=controller.config.get("data_aug_noise", 0.2))
                if warped.shape != (config.get("window_length", 24), len(feature_cols)):
                    logging.error(f"Warped shape mismatch: expected ({config.get('window_length', 24)}, {len(feature_cols)}), got {warped.shape}")
                    continue
                all_windows.append(warped)
                all_targets.append(t_prices)
                all_aux_targets.append(t_volumes)

        all_windows = np.array(all_windows)
        all_targets = np.array(all_targets)
        all_aux_targets = np.array(all_aux_targets)

        if len(all_windows) == 0:
            logging.error("No valid windows generated.")
            tune.report({"loss": float("inf"), "training_iteration": 0})
            return

        if np.any(np.isnan(all_windows)) or np.any(np.isinf(all_windows)):
            logging.error("NaN or Inf found in windows")
            tune.report({"loss": float("inf"), "training_iteration": 0})
            return
        if np.any(np.isnan(all_targets)) or np.any(np.isinf(all_targets)) or np.any(np.isnan(all_aux_targets)) or np.any(np.isinf(all_aux_targets)):
            logging.error("NaN or Inf found in targets")
            tune.report({"loss": float("inf"), "training_iteration": 0})
            return

        all_windows = np.expand_dims(all_windows, axis=2)
        target_scaler = RobustScaler().fit(all_targets)
        aux_target_scaler = RobustScaler().fit(all_aux_targets)
        all_targets = np.clip(target_scaler.transform(all_targets), -10, 10)
        all_aux_targets = np.clip(aux_target_scaler.transform(all_aux_targets), -10, 10)

        split = int(0.8 * len(all_windows))
        train_X, train_y, train_aux_y = all_windows[:split], all_targets[:split], all_aux_targets[:split]
        val_X, val_y, val_aux_y = all_windows[split:], all_targets[split:], all_aux_targets[split:]

        if len(train_X) == 0 or len(val_X) == 0:
            logging.error("Training or validation data is empty.")
            tune.report({"loss": float("inf"), "training_iteration": 0})
            return

        model = build_model_from_config(controller.config, horizons, num_quantiles)
        model.apply(init_weights)
        optimizer = Lion(model.parameters(), lr=controller.config["lr"], weight_decay=controller.config["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

        def self_supervised_pretrain(model, optimizer, windows, temperature=0.5):
            model.train()
            for epoch in range(controller.config.get("ss_epochs", 100)):
                total_loss = 0.0
                for x in windows:
                    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
                    if x_tensor.dim() != 4:
                        x_tensor = x_tensor.unsqueeze(0)
                    aug1 = advanced_time_warp(x_tensor.cpu().numpy(), num_control_points=4, sigma=controller.config.get("data_aug_noise", 0.2))
                    aug2 = advanced_time_warp(x_tensor.cpu().numpy(), num_control_points=4, sigma=controller.config.get("data_aug_noise", 0.2))
                    aug1 = torch.tensor(aug1, dtype=torch.float32).to(device)
                    aug2 = torch.tensor(aug2, dtype=torch.float32).to(device)
                    _, _, emb1 = model(aug1)
                    _, _, emb2 = model(aug2)
                    emb1 = F.normalize(emb1, dim=1)
                    emb2 = F.normalize(emb2, dim=1)
                    sim_matrix = torch.mm(emb1, emb2.t()) / temperature
                    labels = torch.arange(emb1.size(0)).to(device)
                    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(windows)
                scheduler.step(avg_loss)
                logging.info(f"SS Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        ss_windows = train_X[:min(10, len(train_X))]
        self_supervised_pretrain(model, optimizer, ss_windows)

        tasks = [(torch.tensor(train_X[i], dtype=torch.float32).to(device),
                  torch.tensor(train_y[i], dtype=torch.float32).to(device),
                  torch.tensor(train_aux_y[i], dtype=torch.float32).to(device))
                 for i in np.random.choice(len(train_X), min(controller.config["n_tasks"], len(train_X)), replace=False)]
        quantiles_tensor = torch.tensor(quantiles, device=device).view(1, 1, -1)

        def meta_train(model, optimizer, tasks):
            for epoch in range(controller.config.get("meta_epochs", 150)):
                optimizer.zero_grad()
                total_loss = 0.0
                with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    for x, y, aux_y in tasks[:len(tasks)//2]:
                        x = x.unsqueeze(0)
                        price_pred, aux_pred, _ = fmodel(x)
                        price_target = y.unsqueeze(-1).expand(-1, num_quantiles)
                        aux_target = aux_y.unsqueeze(-1).expand(-1, num_quantiles)
                        loss = pinball_loss(price_pred, price_target, quantiles_tensor) + pinball_loss(aux_pred, aux_target, quantiles_tensor)
                        diffopt.step(loss)
                    for x, y, aux_y in tasks[len(tasks)//2:]:
                        x = x.unsqueeze(0)
                        price_pred, aux_pred, _ = fmodel(x)
                        price_target = y.unsqueeze(-1).expand(-1, num_quantiles)
                        aux_target = aux_y.unsqueeze(-1).expand(-1, num_quantiles)
                        loss = pinball_loss(price_pred, price_target, quantiles_tensor) + pinball_loss(aux_pred, aux_target, quantiles_tensor)
                        total_loss += loss
                total_loss = total_loss / max(1, len(tasks) // 2)
                total_loss.backward()
                optimizer.step()
                grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])).item()
                mc_preds = [model(torch.tensor(val_X[0][None], dtype=torch.float32).to(device))[0].cpu().numpy() for _ in range(10)]
                uncertainty = np.mean(np.std(np.array(mc_preds), axis=0))
                dd = run_backtest(df_processed)
                updated_config = controller.update_config({
                    "val_loss": total_loss.item(),
                    "grad_norm": grad_norm,
                    "uncertainty": uncertainty,
                    "train_val_ratio": total_loss.item(),
                    "learning_rates": optimizer.param_groups[0]['lr']
                }, backtest_drawdown=dd, val_X=val_X, val_y=val_y, device=device)

                if controller.arch_changed:
                    model = build_model_from_config(updated_config, horizons, num_quantiles)
                    model.apply(init_weights)
                    optimizer = Lion(model.parameters(), lr=updated_config["lr"], weight_decay=updated_config["weight_decay"])
                    controller.arch_changed = False

                for g in optimizer.param_groups:
                    g['lr'] = updated_config["lr"]

                tune.report({
                    "loss": total_loss.item(),
                    "training_iteration": epoch + 1,
                    "grad_norm": grad_norm,
                    "uncertainty": uncertainty,
                    "curriculum_stage": controller.curriculum_stage,
                    "data_aug_noise": updated_config["data_aug_noise"]
                })

        meta_train(model, optimizer, tasks)

        def supervised_fine_tuning(model, optimizer, train_X, train_y, train_aux_y, sft_epochs=50):
            model.train()
            fisher_matrix = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
            for i in range(min(10, len(train_X))):
                x = torch.tensor(np.expand_dims(train_X[i], axis=0), dtype=torch.float32).to(device)
                y = torch.tensor(train_y[i], dtype=torch.float32).unsqueeze(0).to(device)
                aux_y = torch.tensor(train_aux_y[i], dtype=torch.float32).unsqueeze(0).to(device)
                price_pred, aux_pred, _ = model(x)
                loss = pinball_loss(price_pred, y.unsqueeze(-1).expand(-1, -1, num_quantiles), quantiles_tensor) + \
                       pinball_loss(aux_pred, aux_y.unsqueeze(-1).expand(-1, -1, num_quantiles), quantiles_tensor)
                model.zero_grad()
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        fisher_matrix[n] += p.grad.data ** 2
            ewc_lambda = 1e-3
            for epoch in range(sft_epochs):
                total_loss = 0.0
                for i in range(len(train_X)):
                    x = torch.tensor(np.expand_dims(train_X[i], axis=0), dtype=torch.float32).to(device)
                    y = torch.tensor(train_y[i], dtype=torch.float32).unsqueeze(0).to(device)
                    aux_y = torch.tensor(train_aux_y[i], dtype=torch.float32).unsqueeze(0).to(device)
                    optimizer.zero_grad()
                    price_pred, aux_pred, _ = model(x)
                    price_target = y.unsqueeze(-1).expand(-1, -1, num_quantiles)
                    aux_target = aux_y.unsqueeze(-1).expand(-1, -1, num_quantiles)
                    loss = pinball_loss(price_pred, price_target, quantiles_tensor) + pinball_loss(aux_pred, aux_target, quantiles_tensor)
                    for n, p in model.named_parameters():
                        loss += ewc_lambda * (p - model.state_dict()[n]).pow(2).sum() * fisher_matrix[n]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(train_X)
                scheduler.step(avg_loss)
                logging.info(f"SFT Epoch {epoch+1}/{sft_epochs}, Loss: {avg_loss:.6f}")

        supervised_fine_tuning(model, optimizer, train_X, train_y, train_aux_y, config.get("sft_epochs", 50))

        val_losses, mc_uncertainties = [], []
        for i in range(len(val_X)):
            x_val = torch.tensor(np.expand_dims(val_X[i], axis=0), dtype=torch.float32).to(device)
            y_val = torch.tensor(val_y[i], dtype=torch.float32).unsqueeze(0).to(device)
            aux_y_val = torch.tensor(val_aux_y[i], dtype=torch.float32).unsqueeze(0).to(device)
            preds = [model(x_val)[0].cpu().numpy() for _ in range(10)]
            uncertainty = np.mean(np.std(np.array(preds), axis=0))
            mc_uncertainties.append(uncertainty)
            price_pred, aux_pred, _ = model(x_val)
            price_target = y_val.unsqueeze(-1).expand(-1, -1, num_quantiles)
            aux_target = aux_y_val.unsqueeze(-1).expand(-1, -1, num_quantiles)
            price_loss = pinball_loss(price_pred, price_target, quantiles_tensor)
            aux_loss = pinball_loss(aux_pred, aux_target, quantiles_tensor)
            loss = price_loss + aux_loss
            val_losses.append(loss.item())

        final_loss = np.mean(val_losses) if val_losses else float("inf")
        tune.report({"loss": final_loss, "training_iteration": controller.config.get("meta_epochs", 150)})

    except Exception as e:
        logging.error(f"Error in training: {e}")
        tune.report({"loss": float("inf"), "training_iteration": 0})