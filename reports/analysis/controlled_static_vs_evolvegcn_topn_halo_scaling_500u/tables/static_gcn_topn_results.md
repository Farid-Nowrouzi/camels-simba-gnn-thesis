# Static GCN Top-N results

U500 Top100/Top200/Top500, sample-local min-max normalization, periodic kNN k=8, h32, mean graph pooling, three matched seeds; model-stratified analysis.

| model | Top-N | n | mean MAE | SD MAE | median MAE | mean RMSE | SD RMSE | mean R² | SD R² | mean prediction-SD ratio | SD prediction-SD ratio | mean exact-repeat fraction | maximum exact-repeat fraction | undefined Pearson |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Static GCN | 100 | 3 | 0.096262 | 0.008340 | 0.097587 | 0.112127 | 0.008539 | -0.013416 | 0.014849 | 0.043305 | 0.027620 | 0.448889 | 0.920000 | 0 |
| Static GCN | 200 | 3 | 0.096985 | 0.009836 | 0.098277 | 0.112920 | 0.010053 | -0.026294 | 0.015036 | 0.070753 | 0.067195 | 0.466667 | 0.986667 | 1 |
| Static GCN | 500 | 3 | **0.096039** | 0.008923 | 0.097980 | **0.111753** | 0.009337 | **-0.005794** | 0.005323 | 0.063510 | 0.076279 | 0.462222 | 0.986667 | 1 |
