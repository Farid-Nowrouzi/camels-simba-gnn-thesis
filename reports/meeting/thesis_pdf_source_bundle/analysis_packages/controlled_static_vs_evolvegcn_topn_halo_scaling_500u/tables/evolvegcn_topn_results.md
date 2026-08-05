# EvolveGCN-H Top-N results

U500 Top100/Top200/Top500, sample-local min-max normalization, periodic kNN k=8, h32, mean graph pooling, three matched seeds; model-stratified analysis.

| model | Top-N | n | mean MAE | SD MAE | median MAE | mean RMSE | SD RMSE | mean R² | SD R² | mean prediction-SD ratio | SD prediction-SD ratio | mean exact-repeat fraction | maximum exact-repeat fraction | undefined Pearson |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EvolveGCN-H | 100 | 3 | 0.096962 | 0.009492 | 0.099245 | 0.113241 | 0.009904 | -0.032351 | 0.019526 | 0.062896 | 0.066830 | 0.000000 | 0.000000 | 0 |
| EvolveGCN-H | 200 | 3 | 0.092174 | 0.008329 | 0.094568 | 0.109696 | 0.008587 | 0.025709 | 0.114332 | 0.394002 | 0.075757 | 0.000000 | 0.000000 | 0 |
| EvolveGCN-H | 500 | 3 | **0.086166** | 0.004255 | 0.088085 | **0.102848** | 0.003542 | **0.141301** | 0.093991 | 0.494102 | 0.114047 | 0.000000 | 0.000000 | 0 |
