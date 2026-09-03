"""Train the frozen Notebook-16 GCN-GRU configuration."""

import argparse

from src.models.gcn_gru import GCNGRURegressor
from src.training.temporal_architecture_common import train_from_config


def build_model(config):
    return GCNGRURegressor(
        input_dim=config["node_features"], hidden_dim=config["hidden_dim"],
        num_gcn_layers=config["num_gcn_layers"], dropout=config["dropout"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_from_config(args.config, "GCNGRURegressor", build_model)


if __name__ == "__main__":
    main()
