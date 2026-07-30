# Compatibility Report: Controlled EvolveGCN-H Graph-Pooling Ablation at U750 Top1000

Verdict: **PASS**

Controlled five-seed within-model graph-pooling ablation with exact paired splits.

## Required settings

| field | required_value |
|---|---|
| fixed_scientific_settings.config.num_total_universes | 750 |
| fixed_scientific_settings.config.num_nodes | 1000 |
| fixed_scientific_settings.config.hidden_dim | 32 |
| fixed_scientific_settings.config.num_layers | 2 |
| fixed_scientific_settings.config.batch_size | 4 |
| fixed_scientific_settings.config.temporal_pooling | mean |
| fixed_scientific_settings.config.head_type | linear |
| fixed_scientific_settings.dataset_metadata.num_universes_successful | 750 |
| fixed_scientific_settings.dataset_metadata.num_nodes | 1000 |
| fixed_scientific_settings.dataset_metadata.normalization | none |
| fixed_scientific_settings.dataset_metadata.graph_mode | knn |
| fixed_scientific_settings.dataset_metadata.k | 8 |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | True |
| fixed_scientific_settings.dataset_metadata.box_size | 25.0 |

## Matched settings

| field | status |
|---|---|
| fixed_scientific_settings.config.activation | matched |
| fixed_scientific_settings.config.add_self_loops | matched |
| fixed_scientific_settings.config.batch_size | matched |
| fixed_scientific_settings.config.dropout | matched |
| fixed_scientific_settings.config.epochs | matched |
| fixed_scientific_settings.config.grad_clip_norm | matched |
| fixed_scientific_settings.config.head_type | matched |
| fixed_scientific_settings.config.hidden_dim | matched |
| fixed_scientific_settings.config.learning_rate | matched |
| fixed_scientific_settings.config.model | matched |
| fixed_scientific_settings.config.node_features | matched |
| fixed_scientific_settings.config.normalize_target | matched |
| fixed_scientific_settings.config.num_layers | matched |
| fixed_scientific_settings.config.num_nodes | matched |
| fixed_scientific_settings.config.num_snapshots | matched |
| fixed_scientific_settings.config.num_test_universes | matched |
| fixed_scientific_settings.config.num_total_universes | matched |
| fixed_scientific_settings.config.num_train_universes | matched |
| fixed_scientific_settings.config.num_val_universes | matched |
| fixed_scientific_settings.config.patience | matched |
| fixed_scientific_settings.config.summary_feature_dim | matched |
| fixed_scientific_settings.config.temporal_pooling | matched |
| fixed_scientific_settings.config.test_ratio | matched |
| fixed_scientific_settings.config.train_ratio | matched |
| fixed_scientific_settings.config.use_summary_features | matched |
| fixed_scientific_settings.config.val_ratio | matched |
| fixed_scientific_settings.config.weight_decay | matched |
| fixed_scientific_settings.dataset_metadata.box_size | matched |
| fixed_scientific_settings.dataset_metadata.dataset_type | matched |
| fixed_scientific_settings.dataset_metadata.feature_names | matched |
| fixed_scientific_settings.dataset_metadata.graph_mode | matched |
| fixed_scientific_settings.dataset_metadata.graph_positions | matched |
| fixed_scientific_settings.dataset_metadata.k | matched |
| fixed_scientific_settings.dataset_metadata.node_selection | matched |
| fixed_scientific_settings.dataset_metadata.normalization | matched |
| fixed_scientific_settings.dataset_metadata.num_nodes | matched |
| fixed_scientific_settings.dataset_metadata.num_snapshots | matched |
| fixed_scientific_settings.dataset_metadata.num_universes_failed | matched |
| fixed_scientific_settings.dataset_metadata.num_universes_requested | matched |
| fixed_scientific_settings.dataset_metadata.num_universes_successful | matched |
| fixed_scientific_settings.dataset_metadata.periodic_boundary | matched |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | matched |
| fixed_scientific_settings.dataset_metadata.preprocessing_version | matched |
| fixed_scientific_settings.dataset_metadata.raw_dir | matched |

## Intentional differences

None.

## Unexpected differences

None.

## Split compatibility

Not applicable for a single-family analysis.

Raw saved MAE/RMSE/MSE and prediction-derived metrics were checked by the existing family verifier at the configured numerical tolerance.
