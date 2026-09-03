"""Train one frozen, H-matched EvolveGCN-O production configuration."""

import argparse

from src.models.evolvegcn_o import EvolveGCNORegressor
from src.training.temporal_architecture_common import train_from_config


FROZEN_O_VALUES = {
    "batch_size": 4,
    "epochs": 300,
    "patience": 40,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.2,
    "activation": "relu",
    "temporal_pooling": "mean",
    "graph_pooling": "mean",
    "head_type": "linear",
    "add_self_loops": True,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,
    "grad_clip_norm": 1.0,
    "loss": "MSELoss",
    "scheduler": {
        "name": "ReduceLROnPlateau", "mode": "min", "factor": 0.5,
        "patience": 10, "min_lr": 1e-6,
    },
    "target": "Omega_m",
    "target_normalization": "none",
    "node_normalization": "none",
}


def build_model(config):
    return EvolveGCNORegressor(
        node_features=config["node_features"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        activation=config["activation"],
        temporal_pooling=config["temporal_pooling"],
        graph_pooling=config["graph_pooling"],
        head_type=config["head_type"],
        add_self_loops=config["add_self_loops"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_from_config(
        args.config,
        "EvolveGCNORegressor",
        build_model,
        frozen_values=FROZEN_O_VALUES,
    )


if __name__ == "__main__":
    main()
