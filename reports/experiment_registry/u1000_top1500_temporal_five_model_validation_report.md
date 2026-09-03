# Temporal five-model validation report

## Frozen conclusion

**ROBUST FIVE-MODEL WINNER: GCN-GRU**

Selection uses validation MAE only. Calibration orientation: `predicted = intercept + slope * true`; SD uses n − 1. Test artifacts were checked only structurally; numerical test performance was not computed or used.

## Fairness

| comparison | fairness_level | matched | difference | limitation |
| --- | --- | --- | --- | --- |
| GCN-GRU vs GCN-Transformer | strongest controlled comparison | data; graph; snapshots; features; target; splits; seeds; GCN encoder; pooling; budget; optimizer; scheduler; stopping; clipping | GRU recurrence vs Transformer attention/readout | 11,617 vs 13,825 parameters |
| EvolveGCN-H vs EvolveGCN-O | controlled EvolveGCN-family comparison | data; graph; snapshots; features; target; splits; seeds; h32/l2; pooling; linear head; training protocol | H vs O weight evolution | 3,408,097 vs 11,527 parameters |
| all five | matched-data / matched-split architecture benchmark | U1000/Top1500/raw7/k8; target; splits; seeds | architectures/capacities; Static final snapshot only | not a parameter-matched ablation; architecture-specific details |

## Seed-wise validation

| model | parameters | seed | mae | rmse | r2 | bias | prediction_sd | target_sd | sd_ratio | calibration_slope | calibration_intercept | pearson_r |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static GCN | 5281 | 42 | 0.036552 | 0.049703 | 0.804669 | -0.002862 | 0.100566 | 0.113032 | 0.889707 | 0.798448 | 0.060009 | 0.897427 |
| Static GCN | 5281 | 123 | 0.041268 | 0.049830 | 0.816708 | 0.005160 | 0.098136 | 0.116984 | 0.838891 | 0.761205 | 0.077511 | 0.907395 |
| Static GCN | 5281 | 2025 | 0.036248 | 0.045682 | 0.851369 | 0.004731 | 0.106138 | 0.119096 | 0.891204 | 0.823604 | 0.058254 | 0.924148 |
| EvolveGCN-H | 3408097 | 42 | 0.056782 | 0.066934 | 0.645759 | 0.010913 | 0.092462 | 0.113032 | 0.818017 | 0.662164 | 0.116295 | 0.809475 |
| EvolveGCN-H | 3408097 | 123 | 0.052325 | 0.064374 | 0.694103 | -0.007642 | 0.100742 | 0.116984 | 0.861167 | 0.720011 | 0.077191 | 0.836087 |
| EvolveGCN-H | 3408097 | 2025 | 0.048440 | 0.059299 | 0.749554 | 0.003562 | 0.105646 | 0.119096 | 0.887073 | 0.768677 | 0.073751 | 0.866533 |
| EvolveGCN-O | 11527 | 42 | 0.037125 | 0.046373 | 0.829965 | 0.003924 | 0.100658 | 0.113032 | 0.890528 | 0.812111 | 0.062533 | 0.911944 |
| EvolveGCN-O | 11527 | 123 | 0.031740 | 0.038925 | 0.888157 | 0.013528 | 0.112253 | 0.116984 | 0.959562 | 0.911213 | 0.040429 | 0.949613 |
| EvolveGCN-O | 11527 | 2025 | 0.032799 | 0.043778 | 0.863501 | 0.005274 | 0.113226 | 0.119096 | 0.950714 | 0.884669 | 0.040268 | 0.930532 |
| GCN-GRU | 11617 | 42 | 0.030202 | 0.038229 | 0.884444 | 0.005800 | 0.102577 | 0.113032 | 0.907504 | 0.855334 | 0.050926 | 0.942512 |
| GCN-GRU | 11617 | 123 | 0.027756 | 0.034693 | 0.911154 | 0.003936 | 0.105339 | 0.116984 | 0.900458 | 0.861561 | 0.045881 | 0.956803 |
| GCN-GRU | 11617 | 2025 | 0.026236 | 0.034253 | 0.916437 | 0.005123 | 0.113272 | 0.119096 | 0.951103 | 0.911452 | 0.031991 | 0.958310 |
| GCN-Transformer | 13825 | 42 | 0.032294 | 0.039990 | 0.873550 | 0.006036 | 0.097961 | 0.113032 | 0.866665 | 0.813770 | 0.064127 | 0.938967 |
| GCN-Transformer | 13825 | 123 | 0.035265 | 0.042606 | 0.866004 | 0.001530 | 0.091443 | 0.116984 | 0.781672 | 0.738594 | 0.080732 | 0.944890 |
| GCN-Transformer | 13825 | 2025 | 0.035537 | 0.043750 | 0.863678 | -0.001432 | 0.106401 | 0.119096 | 0.893408 | 0.831001 | 0.049847 | 0.930147 |

## Aggregate validation

| model | parameters | mae_mean | mae_sample_sd | rmse_mean | rmse_sample_sd | r2_mean | r2_sample_sd | bias_mean | sd_ratio_mean | calibration_slope_mean | pearson_r_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static GCN | 5281 | 0.038023 | 0.002815 | 0.048405 | 0.002359 | 0.824248 | 0.024246 | 0.002343 | 0.873267 | 0.794419 | 0.909657 |
| EvolveGCN-H | 3408097 | 0.052516 | 0.004175 | 0.063536 | 0.003886 | 0.696472 | 0.051938 | 0.002278 | 0.855419 | 0.716951 | 0.837365 |
| EvolveGCN-O | 11527 | 0.033888 | 0.002853 | 0.043025 | 0.003781 | 0.860541 | 0.029209 | 0.007575 | 0.933601 | 0.869331 | 0.930696 |
| GCN-GRU | 11617 | 0.028065 | 0.002001 | 0.035725 | 0.002180 | 0.904012 | 0.017150 | 0.004953 | 0.919688 | 0.876115 | 0.952542 |
| GCN-Transformer | 13825 | 0.034365 | 0.001799 | 0.042115 | 0.001927 | 0.867744 | 0.005161 | 0.002045 | 0.847248 | 0.794455 | 0.938001 |

## Paired deltas

Negative means the first named treatment is better.

| comparison | seed | reference_mae | first_model_mae | delta_mae | delta_percent_of_reference |
| --- | --- | --- | --- | --- | --- |
| EvolveGCN-H - Static GCN | 42 | 0.036552 | 0.056782 | 0.020230 | 55.346149 |
| EvolveGCN-H - Static GCN | 123 | 0.041268 | 0.052325 | 0.011057 | 26.793242 |
| EvolveGCN-H - Static GCN | 2025 | 0.036248 | 0.048440 | 0.012192 | 33.635187 |
| EvolveGCN-O - Static GCN | 42 | 0.036552 | 0.037125 | 0.000573 | 1.566916 |
| EvolveGCN-O - Static GCN | 123 | 0.041268 | 0.031740 | -0.009527 | -23.086765 |
| EvolveGCN-O - Static GCN | 2025 | 0.036248 | 0.032799 | -0.003449 | -9.514768 |
| GCN-GRU - Static GCN | 42 | 0.036552 | 0.030202 | -0.006350 | -17.371758 |
| GCN-GRU - Static GCN | 123 | 0.041268 | 0.027756 | -0.013511 | -32.740442 |
| GCN-GRU - Static GCN | 2025 | 0.036248 | 0.026236 | -0.010012 | -27.620731 |
| GCN-Transformer - Static GCN | 42 | 0.036552 | 0.032294 | -0.004258 | -11.648240 |
| GCN-Transformer - Static GCN | 123 | 0.041268 | 0.035265 | -0.006003 | -14.546859 |
| GCN-Transformer - Static GCN | 2025 | 0.036248 | 0.035537 | -0.000711 | -1.960355 |
| EvolveGCN-O - EvolveGCN-H | 42 | 0.056782 | 0.037125 | -0.019657 | -34.618968 |
| EvolveGCN-O - EvolveGCN-H | 123 | 0.052325 | 0.031740 | -0.020584 | -39.339642 |
| EvolveGCN-O - EvolveGCN-H | 2025 | 0.048440 | 0.032799 | -0.015641 | -32.289366 |
| GCN-Transformer - GCN-GRU | 42 | 0.030202 | 0.032294 | 0.002092 | 6.926830 |
| GCN-Transformer - GCN-GRU | 123 | 0.027756 | 0.035265 | 0.007508 | 27.049810 |
| GCN-Transformer - GCN-GRU | 2025 | 0.026236 | 0.035537 | 0.009301 | 35.452661 |

## Seven scientific questions

| question | answer |
| --- | --- |
| Does any temporal model improve over Static GCN? | YES. GCN-GRU and GCN-Transformer improve for all matched seeds; EvolveGCN-O is seed-dependent; EvolveGCN-H degrades for all seeds. |
| Does EvolveGCN-O improve over EvolveGCN-H? | YES — ROBUST IMPROVEMENT across all three seeds. |
| Do GRU or Transformer improve over the EvolveGCN family? | GCN-GRU is lower-MAE than both Evolve variants for every seed. Transformer is lower than H for every seed, but its comparison with O is mixed by seed. |
| GRU versus Transformer? | GCN-GRU has lower validation MAE for all three seeds; Transformer minus GRU is a ROBUST DEGRADATION. |
| Which model has best validation mean MAE? | GCN-GRU. |
| Is the numerical best seed-stable? | YES. GCN-GRU wins all three seeds and also has low across-seed MAE SD. |
| Does temporal modeling reduce prediction compression? | GCN-GRU and EvolveGCN-O move both mean SD ratio and calibration slope closer to 1 than Static. Transformer does not improve SD ratio and has essentially the same slope; H worsens both. Absolute changes are reported without inventing a materiality threshold. |

## Interpretation

Mean ranking: ['GCN-GRU', 'EvolveGCN-O', 'GCN-Transformer', 'Static GCN', 'EvolveGCN-H']. Per-seed winners: {'42': 'GCN-GRU', '123': 'GCN-GRU', '2025': 'GCN-GRU'}. Paired classifications: {'EvolveGCN-H - Static GCN': 'ROBUST DEGRADATION', 'EvolveGCN-O - Static GCN': 'SEED-DEPENDENT', 'GCN-GRU - Static GCN': 'ROBUST IMPROVEMENT', 'GCN-Transformer - Static GCN': 'ROBUST IMPROVEMENT', 'EvolveGCN-O - EvolveGCN-H': 'ROBUST IMPROVEMENT', 'GCN-Transformer - GCN-GRU': 'ROBUST DEGRADATION'}. No n=3 significance test or invented practical threshold is used.

## Limitations

Three screening seeds; Static uses only the final snapshot; capacities are unmatched; EvolveGCN-H is dramatically larger; GRU/Transformer is the tightest comparison; H/O has inherent formulation/capacity differences; no extensive sweep; conclusions are U1000/Top1500/raw7/k8-specific; raw coordinates retain the Notebook 15 symmetry issue; absence of improvement constrains tested mechanisms, not temporal information itself.

Held-out test metrics read before freeze: **NO**. Test used for selection: **NO**.
