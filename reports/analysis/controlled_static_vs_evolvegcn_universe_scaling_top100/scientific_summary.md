# Scientific Summary: Controlled Universe-Count Scaling: Static GCN versus EvolveGCN-H

## Purpose

How does increasing the number of universes from 20 to 500 affect Omega_m regression performance, and how do Static GCN and EvolveGCN-H compare under the controlled Top100 protocol?

## Controlled variables

Universe counts, Top100 graph protocol, minmax normalization, periodic kNN with k=8 and box size 25, seven node features, seeds, and exact split IDs are matched. Error bars are sample standard deviations across the required seeds; test samples are not pooled.

## Intentional differences

Matched universe counts, graph protocol, seeds, and split IDs, with intentional architecture and temporal-input differences.

- `fixed_scientific_settings.config.batch_size`: Static GCN uses its established batch size 8; EvolveGCN-H uses canonical batch size 4.
- `fixed_scientific_settings.config.conv_type`: The explicit Static GCN convolution selector is model-specific.
- `fixed_scientific_settings.config.dataset_format`: Static GCN consumes native final-snapshot graphs; EvolveGCN-H consumes temporal graphs.
- `fixed_scientific_settings.config.head_type`: Each architecture uses its established canonical MLP regression head.
- `fixed_scientific_settings.config.model`: Static GCN and EvolveGCN-H are intentionally different architectures.
- `fixed_scientific_settings.config.num_snapshots`: EvolveGCN-H uses five snapshots; Static GCN uses only the final snapshot.
- `fixed_scientific_settings.config.temporal_pooling`: Temporal pooling applies only to EvolveGCN-H.
- `fixed_scientific_settings.dataset_metadata.dataset_type`: Static metadata identifies native final-snapshot graphs.
- `fixed_scientific_settings.dataset_metadata.node_selection`: Static metadata records the shared Top-N selection explicitly; the temporal family documents it through preprocessing provenance.
- `fixed_scientific_settings.dataset_metadata.num_snapshots`: EvolveGCN-H uses five snapshots; Static GCN uses one final snapshot.
- `fixed_scientific_settings.dataset_metadata.preferred_snapshot`: The final-snapshot value applies to the native Static dataset.
- `fixed_scientific_settings.dataset_metadata.preprocessing_version`: The Static metadata records its preprocessing version explicitly.

## Numerical trends

- EvolveGCN-H: mean test MAE decreased from 0.100515 at 20 universes to 0.096962 at 500 universes. The sequence was non-monotonic.
- Static GCN: mean test MAE decreased from 0.256783 at 20 universes to 0.096883 at 500 universes. The sequence was non-monotonic.

Across 15 paired universe/seed rows, Static GCN had lower MAE in 6, EvolveGCN-H in 9, with 0 exact ties. This is descriptive evidence, not a causal test.

Negative R² values are retained in every table and figure.

## Prediction-collapse diagnostics

Prediction-SD/target-SD ratios ranged from 0.018 to 1.025; the maximum exact repeated-prediction fraction was 0.027. Low dispersion can indicate regression toward the mean, but these diagnostics alone do not establish a collapse mechanism.

## Representative runs

Policy: `median_test_mae`. The best seed is not selected automatically under the default policy.

| model | universes | seed | test_mae |
|---|---|---|---|
| EvolveGCN-H | 20 | 2025 | 0.1010226458311081 |
| EvolveGCN-H | 50 | 123 | 0.10247249715030193 |
| EvolveGCN-H | 100 | 2025 | 0.08814333279927572 |
| EvolveGCN-H | 200 | 42 | 0.09758373474081357 |
| EvolveGCN-H | 500 | 123 | 0.09924482007821402 |
| Static GCN | 20 | 42 | 0.17516184722383818 |
| Static GCN | 50 | 123 | 0.10015647299587727 |
| Static GCN | 100 | 2025 | 0.09527274767557779 |
| Static GCN | 200 | 42 | 0.1013598270714283 |
| Static GCN | 500 | 123 | 0.09737441301345826 |

## Limitations and interpretation

The comparison is controlled for data scale, graph construction, seeds, and splits, but architecture, temporal input, model-specific head, and batch size may intentionally differ. Consequently, observed differences should be interpreted as performance of the complete canonical model protocols, not as an isolated causal effect of temporal modeling or any single architectural component. The five universe-count datasets change population size, so adjacent points are not independent repeated samples of one fixed dataset.
