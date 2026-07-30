# Controlled progression

| stage | fixed_protocol | change | mean_mae | paired_delta |
|---|---|---|---|---|
| Baseline | U750 Top1000; graph mean; temporal mean; all encoder/training settings fixed | MLP head | 0.061404 | — |
| Head ablation | identical to baseline | linear head | 0.055843 | -0.005561 |
