# Compatibility Report: Controlled Node-Feature Normalization Ablation: Static GCN and EvolveGCN-H at U500 Top500

Verdict: **PASS**

Controlled within each model for normalization. Static-versus-Evolve comparisons are descriptive because temporal input, architecture, depth, batch size, pooling availability, and heads differ.

## Required settings

| field | required_value |
|---|---|
| fixed_scientific_settings.config.num_total_universes | 500 |
| fixed_scientific_settings.config.num_nodes | 500 |
| fixed_scientific_settings.config.hidden_dim | 32 |
| fixed_scientific_settings.dataset_metadata.num_universes_successful | 500 |
| fixed_scientific_settings.dataset_metadata.num_nodes | 500 |
| fixed_scientific_settings.dataset_metadata.graph_mode | knn |
| fixed_scientific_settings.dataset_metadata.k | 8 |
| fixed_scientific_settings.dataset_metadata.periodic_boundary | True |
| fixed_scientific_settings.dataset_metadata.periodic_boundary_knn | True |
| fixed_scientific_settings.dataset_metadata.box_size | 25.0 |
| fixed_scientific_settings.dataset_metadata.feature_names | ["log10_Mvir", "X", "Y", "Z", "VX", "VY", "VZ"] |
| fixed_scientific_settings.dataset_metadata.node_selection | top_num_nodes_by_raw_Mvir_descending |
| fixed_scientific_settings.dataset_metadata.graph_positions | raw_physical_XYZ_before_feature_normalization |

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
| fixed_scientific_settings.config.num_nodes | matched |
| fixed_scientific_settings.config.num_total_universes | matched |
| fixed_scientific_settings.config.patience | matched |
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

| field | reason |
|---|---|
| fixed_scientific_settings.config.batch_size | Established model-specific protocols use batch sizes 4 and 8. |
| fixed_scientific_settings.config.conv_type | Convolution mechanics are model-specific. |
| fixed_scientific_settings.config.dataset_format | Evolve uses temporal sequences; Static consumes the final snapshot. |
| fixed_scientific_settings.config.head_type | Regression heads are model-specific. |
| fixed_scientific_settings.config.model | Model-stratified analysis intentionally includes two architectures. |
| fixed_scientific_settings.config.num_layers | Established protocols use Evolve L2 and Static L3. |
| fixed_scientific_settings.config.num_snapshots | Evolve uses five snapshots; Static uses the final snapshot. |
| fixed_scientific_settings.config.temporal_pooling | Mean temporal pooling applies only to Evolve. |

## Unexpected differences

None.

## Split compatibility

PASS: ordered train/validation/test split signatures match for every model, grouping value, and seed.

Raw saved MAE/RMSE/MSE and prediction-derived metrics were checked by the existing family verifier at the configured numerical tolerance.
