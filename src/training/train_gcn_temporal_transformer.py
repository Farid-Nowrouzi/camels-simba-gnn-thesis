"""Train the frozen Notebook-16 GCN temporal-Transformer configuration."""

import argparse

from src.models.gcn_temporal_transformer import GCNTemporalTransformerRegressor
from src.training.temporal_architecture_common import train_from_config


def build_model(config):
    return GCNTemporalTransformerRegressor(
        input_dim=config["node_features"], hidden_dim=config["hidden_dim"],
        num_gcn_layers=config["num_gcn_layers"], dropout=config["dropout"],
        scale_factors=config["scale_factors"], nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        dim_feedforward=config["dim_feedforward"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_from_config(args.config, "GCNTemporalTransformerRegressor", build_model)


if __name__ == "__main__":
    main()
