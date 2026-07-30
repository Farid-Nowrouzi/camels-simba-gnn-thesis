# Compatibility Report: Controlled kNN by Hidden-Dimension Factorial at 500 Universes and Top100 Halos

Verdict: **PASS**

Controlled within each model for k and hidden dimension. Static-versus-Evolve results are descriptive protocol-level comparisons because temporal input, architecture, layer count, batch size, and regression heads intentionally differ.

## Required settings

| field | required_value |
|---|---|
| fixed_scientific_settings.config.num_total_universes | 500 |
| fixed_scientific_settings.config.num_nodes | 100 |
| fixed_scientific_settings.dataset_metadata.num_universes_successful | 500 |
| fixed_scientific_settings.dataset_metadata.num_nodes | 100 |
| fixed_scientific_settings.dataset_metadata.normalization | minmax |
| fixed_scientific_settings.dataset_metadata.graph_mode | knn |
| fixed_scientific_settings.dataset_metadata.periodic_boundary | True |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | True |
| fixed_scientific_settings.dataset_metadata.box_size | 25.0 |
| fixed_scientific_settings.dataset_metadata.feature_names | ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"] |
| fixed_scientific_settings.dataset_metadata.node_selection | top_num_nodes_by_raw_Mvir_descending |

## Matched settings

| field | status |
|---|---|
| fixed_scientific_settings.config.activation | matched |
| fixed_scientific_settings.config.add_self_loops | matched |
| fixed_scientific_settings.config.dropout | matched |
| fixed_scientific_settings.config.epochs | matched |
| fixed_scientific_settings.config.grad_clip_norm | matched |
| fixed_scientific_settings.config.graph_pooling | matched |
| fixed_scientific_settings.config.learning_rate | matched |
| fixed_scientific_settings.config.node_features | matched |
| fixed_scientific_settings.config.normalize_target | matched |
| fixed_scientific_settings.config.num_nodes | matched |
| fixed_scientific_settings.config.num_total_universes | matched |
| fixed_scientific_settings.config.patience | matched |
| fixed_scientific_settings.config.test_ratio | matched |
| fixed_scientific_settings.config.train_ratio | matched |
| fixed_scientific_settings.config.use_summary_features | matched |
| fixed_scientific_settings.config.val_ratio | matched |
| fixed_scientific_settings.config.weight_decay | matched |
| fixed_scientific_settings.dataset_metadata.box_size | matched |
| fixed_scientific_settings.dataset_metadata.feature_names | matched |
| fixed_scientific_settings.dataset_metadata.graph_mode | matched |
| fixed_scientific_settings.dataset_metadata.graph_positions | matched |
| fixed_scientific_settings.dataset_metadata.node_selection | matched |
| fixed_scientific_settings.dataset_metadata.normalization | matched |
| fixed_scientific_settings.dataset_metadata.num_nodes | matched |
| fixed_scientific_settings.dataset_metadata.num_universes_failed | matched |
| fixed_scientific_settings.dataset_metadata.num_universes_requested | matched |
| fixed_scientific_settings.dataset_metadata.num_universes_successful | matched |
| fixed_scientific_settings.dataset_metadata.periodic_boundary | matched |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | matched |
| fixed_scientific_settings.dataset_metadata.preprocessing_version | matched |
| fixed_scientific_settings.dataset_metadata.raw_dir | matched |

## Intentional differences

| field | reason |
|---|---|
| fixed_scientific_settings.config.batch_size | The established model-specific protocols use batch sizes 4 and 8. |
| fixed_scientific_settings.config.conv_type | Convolution implementation is model-specific. |
| fixed_scientific_settings.config.dataset_format | EvolveGCN-H uses temporal graph sequences; Static GCN uses final-snapshot static graphs. |
| fixed_scientific_settings.config.head_type | Each architecture retains its established regression head. |
| fixed_scientific_settings.config.model | The analysis intentionally compares the complete canonical EvolveGCN-H and Static GCN protocols. |
| fixed_scientific_settings.config.num_layers | The established protocols use two EvolveGCN-H layers and three Static GCN layers. |
| fixed_scientific_settings.config.num_snapshots | EvolveGCN-H uses five snapshots while Static GCN uses the final snapshot. |
| fixed_scientific_settings.config.temporal_pooling | Mean temporal pooling applies only to EvolveGCN-H. |
| fixed_scientific_settings.dataset_metadata.dataset_type | The dataset representation is temporal for EvolveGCN-H and final-snapshot static for Static GCN. |
| fixed_scientific_settings.dataset_metadata.num_snapshots | Five temporal snapshots apply only to EvolveGCN-H. |
| fixed_scientific_settings.dataset_metadata.preferred_snapshot | Final-snapshot selection metadata applies only to Static GCN. |

## Unexpected differences

None.

## Split compatibility

PASS: ordered train/validation/test split signatures match for every model, grouping value, and seed.

Raw saved MAE/RMSE/MSE and prediction-derived metrics were checked by the existing family verifier at the configured numerical tolerance.
