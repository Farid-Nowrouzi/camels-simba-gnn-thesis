# Compatibility Report: Controlled Universe-Count Scaling: Static GCN versus EvolveGCN-H

Verdict: **PASS**

Matched universe counts, graph protocol, seeds, and split IDs, with intentional architecture and temporal-input differences.

## Required settings

| field | required_value |
|---|---|
| fixed_scientific_settings.config.epochs | 300 |
| fixed_scientific_settings.config.patience | 40 |
| fixed_scientific_settings.config.learning_rate | 0.001 |
| fixed_scientific_settings.config.weight_decay | 1e-05 |
| fixed_scientific_settings.config.hidden_dim | 32 |
| fixed_scientific_settings.config.num_layers | 2 |
| fixed_scientific_settings.config.dropout | 0.2 |
| fixed_scientific_settings.config.graph_pooling | mean |
| fixed_scientific_settings.config.train_ratio | 0.7 |
| fixed_scientific_settings.config.val_ratio | 0.15 |
| fixed_scientific_settings.config.test_ratio | 0.15 |
| fixed_scientific_settings.config.grad_clip_norm | 1.0 |
| fixed_scientific_settings.config.use_summary_features | False |
| fixed_scientific_settings.config.normalize_target | False |
| fixed_scientific_settings.config.num_nodes | 100 |
| fixed_scientific_settings.config.node_features | 7 |
| fixed_scientific_settings.dataset_metadata.num_nodes | 100 |
| fixed_scientific_settings.dataset_metadata.normalization | minmax |
| fixed_scientific_settings.dataset_metadata.graph_mode | knn |
| fixed_scientific_settings.dataset_metadata.k | 8 |
| fixed_scientific_settings.dataset_metadata.periodic_boundary | True |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | True |
| fixed_scientific_settings.dataset_metadata.box_size | 25.0 |
| fixed_scientific_settings.dataset_metadata.feature_names | ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"] |

## Matched settings

| field | status |
|---|---|
| fixed_scientific_settings.config.activation | matched |
| fixed_scientific_settings.config.add_self_loops | matched |
| fixed_scientific_settings.config.dropout | matched |
| fixed_scientific_settings.config.epochs | matched |
| fixed_scientific_settings.config.grad_clip_norm | matched |
| fixed_scientific_settings.config.graph_pooling | matched |
| fixed_scientific_settings.config.hidden_dim | matched |
| fixed_scientific_settings.config.learning_rate | matched |
| fixed_scientific_settings.config.node_features | matched |
| fixed_scientific_settings.config.normalize_target | matched |
| fixed_scientific_settings.config.num_layers | matched |
| fixed_scientific_settings.config.num_nodes | matched |
| fixed_scientific_settings.config.patience | matched |
| fixed_scientific_settings.config.test_ratio | matched |
| fixed_scientific_settings.config.train_ratio | matched |
| fixed_scientific_settings.config.use_summary_features | matched |
| fixed_scientific_settings.config.val_ratio | matched |
| fixed_scientific_settings.config.weight_decay | matched |
| fixed_scientific_settings.dataset_metadata.box_size | matched |
| fixed_scientific_settings.dataset_metadata.feature_names | matched |
| fixed_scientific_settings.dataset_metadata.graph_mode | matched |
| fixed_scientific_settings.dataset_metadata.k | matched |
| fixed_scientific_settings.dataset_metadata.normalization | matched |
| fixed_scientific_settings.dataset_metadata.num_nodes | matched |
| fixed_scientific_settings.dataset_metadata.periodic_boundary | matched |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | matched |

## Intentional differences

| field | reason |
|---|---|
| fixed_scientific_settings.config.batch_size | Static GCN uses its established batch size 8; EvolveGCN-H uses canonical batch size 4. |
| fixed_scientific_settings.config.conv_type | The explicit Static GCN convolution selector is model-specific. |
| fixed_scientific_settings.config.dataset_format | Static GCN consumes native final-snapshot graphs; EvolveGCN-H consumes temporal graphs. |
| fixed_scientific_settings.config.head_type | Each architecture uses its established canonical MLP regression head. |
| fixed_scientific_settings.config.model | Static GCN and EvolveGCN-H are intentionally different architectures. |
| fixed_scientific_settings.config.num_snapshots | EvolveGCN-H uses five snapshots; Static GCN uses only the final snapshot. |
| fixed_scientific_settings.config.temporal_pooling | Temporal pooling applies only to EvolveGCN-H. |
| fixed_scientific_settings.dataset_metadata.dataset_type | Static metadata identifies native final-snapshot graphs. |
| fixed_scientific_settings.dataset_metadata.node_selection | Static metadata records the shared Top-N selection explicitly; the temporal family documents it through preprocessing provenance. |
| fixed_scientific_settings.dataset_metadata.num_snapshots | EvolveGCN-H uses five snapshots; Static GCN uses one final snapshot. |
| fixed_scientific_settings.dataset_metadata.preferred_snapshot | The final-snapshot value applies to the native Static dataset. |
| fixed_scientific_settings.dataset_metadata.preprocessing_version | The Static metadata records its preprocessing version explicitly. |

## Unexpected differences

None.

## Split compatibility

PASS: ordered train/validation/test split signatures match for every model, grouping value, and seed.

Raw saved MAE/RMSE/MSE and prediction-derived metrics were checked by the existing family verifier at the configured numerical tolerance.
