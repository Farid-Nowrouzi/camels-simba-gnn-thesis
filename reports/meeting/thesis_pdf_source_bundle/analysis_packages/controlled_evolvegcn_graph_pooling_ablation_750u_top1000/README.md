# Controlled EvolveGCN-H graph-pooling ablation

This deterministic package verifies and compares ten existing U750 Top1000 mean and mean_max runs. It reads JSON and CSV artifacts only; it never trains, loads checkpoints, or deserializes graph datasets.

Run the generic builder first, then this focused rebuild script, and finally `scripts/validate_analysis_report.py`.
