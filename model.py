import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.utils import add_remaining_self_loops

# Conditional imports for advanced attention mechanisms
try:
    from performer_pytorch import Performer
except ImportError:
    Performer = None

try:
    from linformer import Linformer
except ImportError:
    Linformer = None

# Configuration constants (adjust as per your setup)
window_length_default = 50  # Example sequence length
num_assets_default = 10     # Example number of assets
num_horizons_default = 5    # Example forecast horizons
num_quantiles_default = 3   # Example quantiles (e.g., 5%, 50%, 95%)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TCNBlock(nn.Module):
    """Temporal Convolutional Network Block"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :x.size(2)]  # Crop padding
        if self.downsample is not None:
            x = self.downsample(x)
        return self.dropout(self.relu(out + x))

class QuantileHead(nn.Module):
    """Output head for quantile regression"""
    def __init__(self, input_dim, num_horizons, num_quantiles):
        super(QuantileHead, self).__init__()
        self.linear = nn.Linear(input_dim, num_horizons * num_quantiles)
        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles

    def forward(self, x):
        out = self.linear(x)
        return out.view(-1, self.num_horizons, self.num_quantiles)

class HybridTCNGNNTransformerModel(nn.Module):
    def __init__(self, feature_dim, tcn_channels, tcn_kernel_size, tcn_layers,
                 gnn_out, transformer_hidden, transformer_layers, n_heads,
                 num_horizons=num_horizons_default, num_quantiles=num_quantiles_default,
                 window_length=window_length_default, num_assets=num_assets_default,
                 dropout=0.1, gnn_layer_type="SAGE", sequence_model_type="TCN",
                 attention_type="default", use_state_space=False, state_space_type="S4",
                 ensemble_size=1, dynamic_graph=False, use_lstm=False, use_efficient_edge=True):
       
        super(HybridTCNGNNTransformerModel, self).__init__()
        self.feature_dim = feature_dim
        self.num_horizons = num_horizons
        self.num_quantiles = num_quantiles
        self.window_length = window_length
        self.num_assets = num_assets
        self.ensemble_size = ensemble_size
        self.sequence_model_type = sequence_model_type
        self.attention_type = attention_type
        self.use_state_space = use_state_space
        self.dynamic_graph = dynamic_graph
        self.use_lstm = use_lstm
        self.use_efficient_edge = use_efficient_edge

        # Sequence Model (TCN or S4 placeholder)
        if sequence_model_type == "TCN":
            layers = []
            in_channels = feature_dim
            for i in range(tcn_layers):
                layers.append(TCNBlock(in_channels, tcn_channels, tcn_kernel_size, dilation=2**i))
                in_channels = tcn_channels
            self.sequence_model = nn.Sequential(*layers)
            seq_out_dim = tcn_channels
        elif sequence_model_type == "S4":
            # Placeholder for S4; replace with actual implementation if available
            self.sequence_model = nn.Linear(feature_dim, tcn_channels)  # Simplified placeholder
            seq_out_dim = tcn_channels
        else:
            raise ValueError(f"Unsupported sequence_model_type: {sequence_model_type}")

        # Optional LSTM Layer
        if use_lstm:
            self.lstm = nn.LSTM(input_size=seq_out_dim, hidden_size=seq_out_dim, num_layers=1, batch_first=True)
            lstm_out_dim = seq_out_dim
        else:
            self.lstm = None
            lstm_out_dim = seq_out_dim

        # GNN Layer
        if gnn_layer_type == "SAGE":
            self.gnn = pyg_nn.SAGEConv(lstm_out_dim, gnn_out)
        elif gnn_layer_type == "GCN":
            self.gnn = pyg_nn.GCNConv(lstm_out_dim, gnn_out)
        elif gnn_layer_type == "GAT":
            self.gnn = pyg_nn.GATConv(lstm_out_dim, gnn_out, heads=n_heads, dropout=dropout)
        elif gnn_layer_type == "GraphConv":
            self.gnn = pyg_nn.GraphConv(lstm_out_dim, gnn_out)
        elif gnn_layer_type == "ChebConv":
            self.gnn = pyg_nn.ChebConv(lstm_out_dim, gnn_out, K=2)
        elif gnn_layer_type == "TransformerConv":
            self.gnn = pyg_nn.TransformerConv(lstm_out_dim, gnn_out, heads=8, dropout=dropout)
        else:
            raise ValueError(f"Unsupported gnn_layer_type: {gnn_layer_type}")

        # Transformer with Attention Mechanism
        if attention_type == "default":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=gnn_out, nhead=n_heads, dim_feedforward=transformer_hidden,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        elif attention_type == "performer":
            if Performer is None:
                raise ImportError("Install performer-pytorch for 'performer' attention")
            self.transformer = Performer(
                dim=gnn_out, depth=transformer_layers, heads=n_heads, dim_head=gnn_out // n_heads
            )
        elif attention_type == "linformer":
            if Linformer is None:
                raise ImportError("Install linformer for 'linformer' attention")
            self.transformer = Linformer(
                dim=gnn_out, seq_len=window_length, depth=transformer_layers, heads=n_heads, k=256
            )
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        # State-Space Model (optional layer)
        if use_state_space:
            if state_space_type in ["S4", "S5"]:
                # Placeholder; replace with actual S4/S5 implementation if available
                self.state_space = nn.Linear(gnn_out, gnn_out)
            else:
                raise ValueError(f"Unsupported state_space_type: {state_space_type}")
        else:
            self.state_space = None

        # Ensemble of Output Heads
        self.price_quantile_heads = nn.ModuleList([
            QuantileHead(gnn_out, num_horizons, num_quantiles) for _ in range(ensemble_size)
        ])
        self.aux_quantile_heads = nn.ModuleList([
            QuantileHead(gnn_out, num_horizons, num_quantiles) for _ in range(ensemble_size)
        ])

        # Move model to device
        self.to(device)

    def forward(self, x, edge_index=None, return_attention=False):
     
        # Move input to device
        x = x.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)

        batch_size, seq_len, num_assets, feat_dim = x.size()
        if seq_len != self.window_length or num_assets != self.num_assets or feat_dim != self.feature_dim:
            raise ValueError(f"Input shape mismatch: expected [batch, {self.window_length}, {self.num_assets}, {self.feature_dim}], got {x.shape}")

        # Sequence Processing
        if self.sequence_model_type == "TCN":
            tcn_input = x.view(batch_size * self.num_assets, feat_dim, seq_len)
            seq_out = self.sequence_model(tcn_input)  # [batch_size * num_assets, tcn_channels, seq_len]
            seq_out = seq_out.permute(0, 2, 1)  # [batch_size * num_assets, seq_len, tcn_channels]
        elif self.sequence_model_type == "S4":
            # Placeholder for S4
            seq_out = self.sequence_model(x.view(batch_size * self.num_assets, seq_len, feat_dim))

        # Optional LSTM Layer
        if self.use_lstm:
            seq_out, _ = self.lstm(seq_out)  # [batch_size * num_assets, seq_len, lstm_out_dim]

        # GNN Processing
        seq_out_last = seq_out[:, -1, :]  # [batch_size * num_assets, lstm_out_dim]
        if edge_index is None:
            # Default fully connected graph for each batch
            edge_index = torch.combinations(torch.arange(self.num_assets), r=2).t().to(x.device)
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_assets)
            # Apply GNN per batch to handle batched graphs correctly
            gnn_outs = []
            for b in range(batch_size):
                node_features = seq_out_last[b * self.num_assets:(b + 1) * self.num_assets]
                gnn_out_b = self.gnn(node_features, edge_index)
                gnn_outs.append(gnn_out_b)
            gnn_out = torch.stack(gnn_outs, dim=0)  # [batch_size, num_assets, gnn_out]
        else:
            # Assume provided edge_index is suitable for batched graphs
            gnn_out = self.gnn(seq_out_last, edge_index)

        # Transformer Processing
        transformer_input = gnn_out  # [batch_size, num_assets, gnn_out]
        if self.attention_type == "default":
            transformer_out = self.transformer(transformer_input)
            attn_weights = None  # Attention weights not captured for default Transformer
        else:
            transformer_out = self.transformer(transformer_input)
            attn_weights = None

        # State-Space Processing
        if self.use_state_space:
            transformer_out = self.state_space(transformer_out)

        # Ensemble Predictions
        final_state = transformer_out[:, -1, :]  # [batch_size, gnn_out]
        price_quantiles_list = [head(final_state) for head in self.price_quantile_heads]
        aux_quantiles_list = [head(final_state) for head in self.aux_quantile_heads]
        price_quantiles = torch.mean(torch.stack(price_quantiles_list, dim=0), dim=0)
        aux_quantiles = torch.mean(torch.stack(aux_quantiles_list, dim=0), dim=0)

        if return_attention:
            return price_quantiles, aux_quantiles, attn_weights
        return price_quantiles, aux_quantiles

def pinball_loss(pred, target, quantiles):
    """Pinball loss for quantile regression"""
    errors = target - pred
    loss = torch.max(quantiles * errors, (quantiles - 1) * errors)
    return loss.mean()

# Example instantiation (uncomment and adjust as needed)
# model = HybridTCNGNNTransformerModel(
#     feature_dim=20, tcn_channels=64, tcn_kernel_size=3, tcn_layers=4,
#     gnn_out=128, transformer_hidden=256, transformer_layers=2, n_heads=8,
#     window_length=50, num_assets=10,
#     gnn_layer_type="GAT", sequence_model_type="TCN",
#     attention_type="performer", use_state_space=True, state_space_type="S4",
#     ensemble_size=3, dynamic_graph=False, use_lstm=True, use_efficient_edge=True
# )