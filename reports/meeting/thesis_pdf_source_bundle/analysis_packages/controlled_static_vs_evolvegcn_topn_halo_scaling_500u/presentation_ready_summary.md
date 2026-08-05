# Presentation-ready summary

## Recommended supervisor-meeting sequence

1. Protocol table: define U500 separately from Top-N.
2. Six-row model-stratified main-results table.
3. Test MAE versus Top-N with individual matched seeds.
4. Prediction-SD ratio versus Top-N.
5. Exact repeated-prediction fraction versus Top-N.
6. Median-MAE representative true-versus-predicted panels.
7. Nodes, neighbor selections, and dense adjacency capacity versus Top-N.
8. Conclusion slide.

## Concise conclusion slide

At fixed U500 protocols, increasing Top-N consistently improves EvolveGCN-H
error and prediction dispersion across all three seeds, but dispersion remains
compressed. Static GCN remains nearly flat and highly repetitive. Top500 costs
5× more nodes and 25× dense adjacency capacity than Top100, so the gain is
model-dependent and does not show that graph size alone solves collapse.
