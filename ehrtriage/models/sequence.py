"""
Sequence models for temporal risk prediction.

Implements GRU and Transformer architectures for processing EHR timelines.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from ehrtriage.config import get_default_model_config
from ehrtriage.sequence_builder import SequenceDataset


class GRURiskModel(nn.Module):
    """GRU-based risk prediction model."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Initialize GRU model.

        Args:
            input_dim: Dimension of input features per timestep
            static_dim: Dimension of static features
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
        """
        super().__init__()

        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # GRU layer
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Calculate output dimension
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Combine with static features
        combined_dim = gru_output_dim + static_dim

        # Temporal attention to focus on informative windows
        self.attention = TemporalAttentionPooling(gru_output_dim)

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with mask-aware pooling.

        Args:
            sequence: Input sequence [batch, seq_len, input_dim]
            static: Static features [batch, static_dim]
            mask: Mask for padding [batch, seq_len]

        Returns:
            Risk logits [batch, 1]
        """
        lengths = mask.sum(dim=1).long().clamp(min=1)

        packed_seq = pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed_seq)

        # Recover padded output to apply attention with the original mask
        gru_out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=sequence.size(1)
        )

        attended = self.attention(gru_out, mask)

        # Combine with static features
        combined = torch.cat([attended, static], dim=1)

        # Predict
        logits = self.fc(combined)

        return logits


class TransformerRiskModel(nn.Module):
    """Transformer-based risk prediction model."""

    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ):
        """
        Initialize Transformer model.

        Args:
            input_dim: Dimension of input features per timestep
            static_dim: Dimension of static features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.static_dim = static_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Combine with static features
        combined_dim = d_model + static_dim

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, sequence: torch.Tensor, static: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequence: Input sequence [batch, seq_len, input_dim]
            static: Static features [batch, static_dim]
            mask: Mask for padding [batch, seq_len]

        Returns:
            Risk logits [batch, 1]
        """
        # Project input
        x = self.input_projection(sequence)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask (True for positions to mask)
        src_key_padding_mask = mask == 0  # [batch, seq_len]

        # Transformer encoding
        encoded = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [batch, seq_len, d_model]

        # Pool over sequence (use mean of non-masked positions)
        mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
        mask_sum = mask_expanded.sum(dim=1).clamp(min=1e-6)
        pooled = (encoded * mask_expanded).sum(dim=1) / mask_sum  # [batch, d_model]

        # Combine with static features
        combined = torch.cat([pooled, static], dim=1)

        # Predict
        logits = self.fc(combined)

        return logits


class TemporalAttentionPooling(nn.Module):
    """Applies learnable attention weights over temporal outputs."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden_dim = max(32, input_dim // 2)
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: Tensor [batch, seq_len, input_dim]
            mask: Padding mask [batch, seq_len]
        Returns:
            Weighted context vector [batch, input_dim]
        """
        attn_mask = mask.clone()
        zero_length = attn_mask.sum(dim=1) == 0
        if zero_length.any():
            attn_mask[zero_length, 0] = 1.0

        scores = self.attn(sequence).squeeze(-1)
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return pooled


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def train_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    early_stopping_patience: int = 10,
    label_smoothing: float = 0.05,
    grad_clip: Optional[float] = 5.0,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Train a sequence model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        early_stopping_patience: Patience for early stopping
        label_smoothing: Amount of label smoothing for stability
        grad_clip: Gradient clipping value (None to disable)
        device: Device to train on

    Returns:
        Tuple of (trained model, training history)
    """
    from ehrtriage.evaluation.metrics import compute_classification_metrics

    model = model.to(device)

    amp_enabled = device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    # Class imbalance-aware weighting
    dataset = getattr(train_loader, "dataset", None)
    pos_weight_tensor = None
    if dataset is not None and hasattr(dataset, "labels"):
        labels_tensor = dataset.labels
        pos_count = float(labels_tensor.sum().item())
        neg_count = float(len(labels_tensor) - pos_count)
        if pos_count > 0 and neg_count > 0:
            imbalance = neg_count / pos_count
            pos_weight_tensor = torch.tensor([imbalance], device=device)
            print(f"Using positive class weight: {imbalance:.2f}")

    # Loss and optimizer
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        if pos_weight_tensor is not None
        else nn.BCEWithLogitsLoss()
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = None
    if val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    best_val_auroc = 0.0
    patience_counter = 0
    best_model_state = None

    print(f"Training on {device}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            sequence = batch["sequence"].to(device, non_blocking=True)
            static = batch["static"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(sequence, static, mask).squeeze()
                smooth_labels = (
                    labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
                    if label_smoothing > 0
                    else labels
                )
                loss = criterion(logits, smooth_labels)

            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    sequence = batch["sequence"].to(device, non_blocking=True)
                    static = batch["static"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    labels = batch["label"].float().to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        logits = model(sequence, static, mask).squeeze()
                        smooth_labels = (
                            labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
                            if label_smoothing > 0
                            else labels
                        )
                        loss = criterion(logits, smooth_labels)

                    val_loss += loss.item()

                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            # Compute metrics
            metrics = compute_classification_metrics(
                np.array(all_labels), np.array(all_probs)
            )
            history["val_auroc"].append(metrics["auroc"])
            history["val_auprc"].append(metrics["auprc"])

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val AUROC: {metrics['auroc']:.4f}, "
                f"Val AUPRC: {metrics['auprc']:.4f}"
            )

            # Early stopping
            if metrics["auroc"] > best_val_auroc:
                best_val_auroc = metrics["auroc"]
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if scheduler is not None:
                scheduler.step(metrics["auroc"])

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with AUROC: {best_val_auroc:.4f}")

    return model, history


def save_sequence_model(
    model: nn.Module,
    output_dir: Path,
    model_name: str,
    config: Dict,
    history: Optional[Dict] = None,
) -> None:
    """
    Save sequence model to directory.

    Args:
        model: Model to save
        output_dir: Output directory
        model_name: Name for the model files
        config: Model configuration
        history: Training history (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), output_dir / f"{model_name}.pt")

    # Save config
    config_to_save = {**config}
    if history is not None:
        config_to_save["history"] = history

    with open(output_dir / f"{model_name}_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print(f"Saved {model_name} to {output_dir}")


def load_sequence_model(
    model_class: type,
    input_dir: Path,
    model_name: str,
    device: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Load sequence model from directory.

    Args:
        model_class: Model class (GRURiskModel or TransformerRiskModel)
        input_dir: Input directory
        model_name: Name of the model files
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    input_dir = Path(input_dir)

    # Load config
    with open(input_dir / f"{model_name}_config.json", "r") as f:
        config = json.load(f)

    # Create model instance
    model_params = {
        k: v
        for k, v in config.items()
        if k
        in [
            "input_dim",
            "static_dim",
            "hidden_dim",
            "num_layers",
            "dropout",
            "bidirectional",
            "d_model",
            "nhead",
            "dim_feedforward",
        ]
    }

    model = model_class(**model_params)

    # Load state
    model.load_state_dict(
        torch.load(input_dir / f"{model_name}.pt", map_location=device)
    )
    model = model.to(device)
    model.eval()

    return model, config
