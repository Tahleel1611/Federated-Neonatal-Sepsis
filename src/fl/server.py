from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import flwr as fl

from src.config import FederatedConfig, ModelConfig
from src.constants import MODEL_FEATURE_COLUMNS
from src.models.transformer_lstm import TransformerLSTMSepsisModel


def _weighted_avg(metrics):
    if not metrics:
        return {}

    result = {}
    keys = set().union(*[m.keys() for _, m in metrics])
    total_examples = sum(num_examples for num_examples, _ in metrics)
    for key in keys:
        weighted_sum = 0.0
        has_value = False
        for num_examples, m in metrics:
            if key in m:
                weighted_sum += float(m[key]) * num_examples
                has_value = True
        if has_value and total_examples > 0:
            result[key] = weighted_sum / total_examples
    return result


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args,
        model_output_path: str,
        final_round: int,
        seq_len_steps: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_output_path = str(model_output_path)
        self.final_round = int(final_round)
        self.seq_len_steps = int(seq_len_steps)

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and int(server_round) >= self.final_round:
            self._save_global_checkpoint(aggregated_parameters, server_round)

        return aggregated_parameters, aggregated_metrics

    def _save_global_checkpoint(self, aggregated_parameters, server_round: int) -> None:
        ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
        model_cfg = ModelConfig()

        model = TransformerLSTMSepsisModel(
            input_size=len(MODEL_FEATURE_COLUMNS),
            d_model=model_cfg.d_model,
            num_heads=model_cfg.num_heads,
            transformer_layers=model_cfg.transformer_layers,
            lstm_hidden=model_cfg.lstm_hidden,
            lstm_layers=model_cfg.lstm_layers,
            dropout=model_cfg.dropout,
        )
        base_state = model.state_dict()

        if len(ndarrays) != len(base_state):
            raise ValueError(
                "Flower parameters do not match model state_dict length: "
                f"{len(ndarrays)} vs {len(base_state)}"
            )

        updated_state = OrderedDict()
        for (name, tensor_template), weight_array in zip(base_state.items(), ndarrays):
            updated_state[name] = torch.tensor(weight_array, dtype=tensor_template.dtype)

        model.load_state_dict(updated_state, strict=True)

        out_path = Path(self.model_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_size": int(len(MODEL_FEATURE_COLUMNS)),
                "d_model": model_cfg.d_model,
                "num_heads": model_cfg.num_heads,
                "transformer_layers": model_cfg.transformer_layers,
                "lstm_hidden": model_cfg.lstm_hidden,
                "lstm_layers": model_cfg.lstm_layers,
                "dropout": model_cfg.dropout,
                "seq_len_steps": int(self.seq_len_steps),
                "round": int(server_round),
            },
            out_path,
        )
        print(f"Saved final global model checkpoint: {out_path} (round={server_round})")


def _series_to_dict(series):
    return {int(r): float(v) for r, v in series}


def _build_initial_parameters(initial_positive_prior: float):
    model_cfg = ModelConfig()
    model = TransformerLSTMSepsisModel(
        input_size=len(MODEL_FEATURE_COLUMNS),
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        transformer_layers=model_cfg.transformer_layers,
        lstm_hidden=model_cfg.lstm_hidden,
        lstm_layers=model_cfg.lstm_layers,
        dropout=model_cfg.dropout,
    )
    if hasattr(model, "initialize_output_bias"):
        model.initialize_output_bias(float(initial_positive_prior))
        print(f"[server] Phase 1: Initialized round-0 bias | pos_prior={float(initial_positive_prior):.6f}")
    ndarrays = [value.detach().cpu().numpy() for _, value in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


def _export_round_table(history, output_path: str):
    loss_by_round = _series_to_dict(history.losses_distributed)
    epsilon_by_round = _series_to_dict(history.metrics_distributed_fit.get("epsilon", []))
    auroc_by_round = _series_to_dict(history.metrics_distributed.get("AUROC", []))
    pr_auc_by_round = _series_to_dict(history.metrics_distributed.get("PR_AUC", []))
    sens_by_round = _series_to_dict(history.metrics_distributed.get("Sensitivity", []))
    spec_by_round = _series_to_dict(history.metrics_distributed.get("Specificity", []))

    round_ids = sorted(
        set(loss_by_round)
        | set(epsilon_by_round)
        | set(auroc_by_round)
        | set(pr_auc_by_round)
        | set(sens_by_round)
        | set(spec_by_round)
    )
    cumulative_epsilon = 0.0
    rows = []
    for round_id in round_ids:
        eps = epsilon_by_round.get(round_id, 0.0)
        cumulative_epsilon += eps
        rows.append(
            {
                "round": round_id,
                "epsilon_round": eps,
                "epsilon_cumulative": cumulative_epsilon,
                "distributed_loss": loss_by_round.get(round_id, np.nan),
                "AUROC": auroc_by_round.get(round_id, np.nan),
                "PR_AUC": pr_auc_by_round.get(round_id, np.nan),
                "tuned_sensitivity": sens_by_round.get(round_id, np.nan),
                "tuned_specificity": spec_by_round.get(round_id, np.nan),
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "epsilon_round",
                "epsilon_cumulative",
                "distributed_loss",
                "AUROC",
                "PR_AUC",
                "tuned_sensitivity",
                "tuned_specificity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved privacy-utility rounds table: {out}")


def main(
    server_address: str | None = None,
    output_path: str = "results/fl_round_metrics.csv",
    rounds: int | None = None,
    min_clients: int = 2,
    save_global_model_path: str = "results/federated_global_model_final.pt",
    seq_len_steps: int = 60,
    initial_positive_prior: float = 0.0014,
):
    cfg = FederatedConfig()
    bind_address = server_address or "0.0.0.0:8080"
    cfg.server_address = bind_address
    if rounds is not None:
        cfg.rounds = int(rounds)

    min_clients = max(2, int(min_clients))
    initial_parameters = _build_initial_parameters(initial_positive_prior)

    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        fit_metrics_aggregation_fn=_weighted_avg,
        evaluate_metrics_aggregation_fn=_weighted_avg,
        initial_parameters=initial_parameters,
        model_output_path=save_global_model_path,
        final_round=int(cfg.rounds),
        seq_len_steps=seq_len_steps,
    )

    history = fl.server.start_server(
        server_address=cfg.server_address,
        config=fl.server.ServerConfig(num_rounds=cfg.rounds),
        strategy=strategy,
    )
    _export_round_table(history, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default=None, help="Legacy alias for --bind_address")
    parser.add_argument("--bind_address", type=str, default=None, help="Address Flower binds to (default: 0.0.0.0:8080)")
    parser.add_argument("--output_path", type=str, default="results/fl_round_metrics.csv")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--min_clients", type=int, default=2)
    parser.add_argument("--save_global_model_path", type=str, default="results/federated_global_model_final.pt")
    parser.add_argument("--seq_len_steps", type=int, default=60)
    parser.add_argument("--initial_positive_prior", type=float, default=0.0014)
    args = parser.parse_args()
    main(
        args.bind_address or args.server_address,
        args.output_path,
        args.rounds,
        args.min_clients,
        args.save_global_model_path,
        args.seq_len_steps,
        args.initial_positive_prior,
    )