from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerLSTMSepsisModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        transformer_layers: int = 2,
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.attn_score = nn.Linear(lstm_hidden, 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1),
        )

    def initialize_output_bias(self, positive_prior: float) -> None:
        positive_prior = float(min(max(positive_prior, 1e-6), 1.0 - 1e-6))
        logit_bias = math.log(positive_prior / (1.0 - positive_prior))
        final_linear = self.classifier[-1]
        if isinstance(final_linear, nn.Linear) and final_linear.bias is not None:
            with torch.no_grad():
                final_linear.bias.fill_(logit_bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.input_proj(x)
        global_repr = self.transformer(projected)
        local_repr, _ = self.lstm(global_repr)

        attn_logits = self.attn_score(local_repr)
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(attn_weights * local_repr, dim=1)

        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights.squeeze(-1)