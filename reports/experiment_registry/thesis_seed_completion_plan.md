# Thesis Seed Completion Plan

No training has been run. Commands below are future commands only. Use new experiment names to avoid overwriting existing historical folders.

## Classification Summary

|Family|Classification|Reason|
|---|---|---|
|20U-500U EvolveGCN-H scaling|requires rerunning incompatible seeds|20U/50U/100U seed42 and seed2025 use epochs=200, not final32 epochs=300.|
|20U-500U Static GCN scaling|requires rerunning incompatible seeds|20U/50U/100U seed42 and seed2025 use batch_size=4 and epochs=200, not final32 batch_size=8 and epochs=300.|
|500U Top500 normalization ablation|already sufficient|Five seeds exist; seed42 minmax is compatible despite missing explicit default fields.|
|750U Top1000 head ablation|already sufficient|Five matched seeds exist.|
|750U Top1000 graph-pooling ablation|already sufficient|Five matched mean and mean_max seeds exist.|
|750U Top1000 temporal-pooling ablation|already sufficient|Five matched temporal-last seeds exist.|
|Matched 750U Top1000 Static GCN|recommended missing family|No matched 750U Top1000 Static GCN family exists.|
|Debug/archive/diagnostic families|should not receive more runs|They are not final thesis comparison families.|

## Recommended Additional Runs

Total recommended additional runs before the next thesis-quality comparison table: **17**. This is 5 matched 750U Static GCN runs plus 12 compatibility reruns for the historical 20U/50U/100U scaling families.

|priority|family|universe|missing_or_rerun_seed|reference|dataset_exists|graph_rebuild|storage_cost|training_cost|scientific_value|
|---|---|---|---|---|---|---|---|---|---|
|P0|Matched 750U Top1000 Static GCN final snapshot|750|42|experiments/evolvegcn_h_750u_top1000_h32_seed42_none_linear_head_temporal_last/config.json|True|no|moderate|moderate|very high|
|P0|Matched 750U Top1000 Static GCN final snapshot|750|123|experiments/evolvegcn_h_750u_top1000_h32_seed123_none_linear_head_temporal_last/config.json|True|no|moderate|moderate|very high|
|P0|Matched 750U Top1000 Static GCN final snapshot|750|777|experiments/evolvegcn_h_750u_top1000_h32_seed777_none_linear_head_temporal_last/config.json|True|no|moderate|moderate|very high|
|P0|Matched 750U Top1000 Static GCN final snapshot|750|999|experiments/evolvegcn_h_750u_top1000_h32_seed999_none_linear_head_temporal_last/config.json|True|no|moderate|moderate|very high|
|P0|Matched 750U Top1000 Static GCN final snapshot|750|2025|experiments/evolvegcn_h_750u_top1000_h32_seed2025_none_linear_head_temporal_last/config.json|True|no|moderate|moderate|very high|
|P1|EvolveGCN-H canonical Top100 minmax scaling|20|42|evolvegcn_h_20u_seed123_final32|True|no|low|low|high|
|P1|EvolveGCN-H canonical Top100 minmax scaling|20|2025|evolvegcn_h_20u_seed123_final32|True|no|low|low|high|
|P1|EvolveGCN-H canonical Top100 minmax scaling|50|42|evolvegcn_h_50u_seed123_final32|True|no|low|low|high|
|P1|EvolveGCN-H canonical Top100 minmax scaling|50|2025|evolvegcn_h_50u_seed123_final32|True|no|low|low|high|
|P1|EvolveGCN-H canonical Top100 minmax scaling|100|42|evolvegcn_h_100u_seed123_final32|True|no|low|low|high|
|P1|EvolveGCN-H canonical Top100 minmax scaling|100|2025|evolvegcn_h_100u_seed123_final32|True|no|low|low|high|
|P1|Static GCN canonical Top100 minmax scaling|20|42|static_gcn_20u_seed123_final32|True|no|low|low|high|
|P1|Static GCN canonical Top100 minmax scaling|20|2025|static_gcn_20u_seed123_final32|True|no|low|low|high|
|P1|Static GCN canonical Top100 minmax scaling|50|42|static_gcn_50u_seed123_final32|True|no|low|low|high|
|P1|Static GCN canonical Top100 minmax scaling|50|2025|static_gcn_50u_seed123_final32|True|no|low|low|high|
|P1|Static GCN canonical Top100 minmax scaling|100|42|static_gcn_100u_seed123_final32|True|no|low|low|high|
|P1|Static GCN canonical Top100 minmax scaling|100|2025|static_gcn_100u_seed123_final32|True|no|low|low|high|

## Future Commands: P0 Matched 750U Top1000 Static GCN

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt --dataset_format temporal_final_snapshot --split_config_path experiments/evolvegcn_h_750u_top1000_h32_seed42_none_linear_head_temporal_last/config.json --experiment_name static_gcn_750u_top1000_final_snapshot_h32_seed42_none_matched_temporal_last_split --seed 42 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.6 --val_ratio 0.1333333333 --test_ratio 0.2666666667 --grad_clip_norm 1.0 --device cuda
```

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt --dataset_format temporal_final_snapshot --split_config_path experiments/evolvegcn_h_750u_top1000_h32_seed123_none_linear_head_temporal_last/config.json --experiment_name static_gcn_750u_top1000_final_snapshot_h32_seed123_none_matched_temporal_last_split --seed 123 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.6 --val_ratio 0.1333333333 --test_ratio 0.2666666667 --grad_clip_norm 1.0 --device cuda
```

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt --dataset_format temporal_final_snapshot --split_config_path experiments/evolvegcn_h_750u_top1000_h32_seed777_none_linear_head_temporal_last/config.json --experiment_name static_gcn_750u_top1000_final_snapshot_h32_seed777_none_matched_temporal_last_split --seed 777 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.6 --val_ratio 0.1333333333 --test_ratio 0.2666666667 --grad_clip_norm 1.0 --device cuda
```

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt --dataset_format temporal_final_snapshot --split_config_path experiments/evolvegcn_h_750u_top1000_h32_seed999_none_linear_head_temporal_last/config.json --experiment_name static_gcn_750u_top1000_final_snapshot_h32_seed999_none_matched_temporal_last_split --seed 999 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.6 --val_ratio 0.1333333333 --test_ratio 0.2666666667 --grad_clip_norm 1.0 --device cuda
```

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt --dataset_format temporal_final_snapshot --split_config_path experiments/evolvegcn_h_750u_top1000_h32_seed2025_none_linear_head_temporal_last/config.json --experiment_name static_gcn_750u_top1000_final_snapshot_h32_seed2025_none_matched_temporal_last_split --seed 2025 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 3 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.6 --val_ratio 0.1333333333 --test_ratio 0.2666666667 --grad_clip_norm 1.0 --device cuda
```

## Future Commands: P1 Scaling Compatibility Reruns

### EvolveGCN-H canonical Top100 minmax scaling U20 seed 42
Reference config: `evolvegcn_h_20u_seed123_final32`

```bash
python3 src/training/train_evolvegcn_h.py --dataset_path data/processed/temporal_20u_minmax/camels_20u_temporal_logmass_minmax_top100_periodic_knn.pt --experiment_name evolvegcn_h_20u_seed42_final32_compat --seed 42 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type mlp --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### EvolveGCN-H canonical Top100 minmax scaling U20 seed 2025
Reference config: `evolvegcn_h_20u_seed123_final32`

```bash
python3 src/training/train_evolvegcn_h.py --dataset_path data/processed/temporal_20u_minmax/camels_20u_temporal_logmass_minmax_top100_periodic_knn.pt --experiment_name evolvegcn_h_20u_seed2025_final32_compat --seed 2025 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type mlp --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### EvolveGCN-H canonical Top100 minmax scaling U50 seed 42
Reference config: `evolvegcn_h_50u_seed123_final32`

```bash
python3 src/training/train_evolvegcn_h.py --dataset_path data/processed/temporal_50u_minmax/camels_50u_temporal_logmass_minmax_top100_periodic_knn.pt --experiment_name evolvegcn_h_50u_seed42_final32_compat --seed 42 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type mlp --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### EvolveGCN-H canonical Top100 minmax scaling U50 seed 2025
Reference config: `evolvegcn_h_50u_seed123_final32`

```bash
python3 src/training/train_evolvegcn_h.py --dataset_path data/processed/temporal_50u_minmax/camels_50u_temporal_logmass_minmax_top100_periodic_knn.pt --experiment_name evolvegcn_h_50u_seed2025_final32_compat --seed 2025 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type mlp --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### EvolveGCN-H canonical Top100 minmax scaling U100 seed 42
Reference config: `evolvegcn_h_100u_seed123_final32`

```bash
python3 src/training/train_evolvegcn_h.py --dataset_path data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt --experiment_name evolvegcn_h_100u_seed42_final32_compat --seed 42 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type mlp --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### EvolveGCN-H canonical Top100 minmax scaling U100 seed 2025
Reference config: `evolvegcn_h_100u_seed123_final32`

```bash
python3 src/training/train_evolvegcn_h.py --dataset_path data/processed/temporal_100u_minmax/camels_100u_temporal_logmass_minmax_top100_periodic_knn.pt --experiment_name evolvegcn_h_100u_seed2025_final32_compat --seed 2025 --batch_size 4 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --activation relu --temporal_pooling mean --graph_pooling mean --head_type mlp --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### Static GCN canonical Top100 minmax scaling U20 seed 42
Reference config: `static_gcn_20u_seed123_final32`

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/static_20u_logmass_minmax_top100_periodic_knn/camels_20u_static_logmass_minmax_top100_periodic_knn.pt --experiment_name static_gcn_20u_seed42_final32_compat --seed 42 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### Static GCN canonical Top100 minmax scaling U20 seed 2025
Reference config: `static_gcn_20u_seed123_final32`

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/static_20u_logmass_minmax_top100_periodic_knn/camels_20u_static_logmass_minmax_top100_periodic_knn.pt --experiment_name static_gcn_20u_seed2025_final32_compat --seed 2025 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### Static GCN canonical Top100 minmax scaling U50 seed 42
Reference config: `static_gcn_50u_seed123_final32`

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/static_50u_logmass_minmax_top100_periodic_knn/camels_50u_static_logmass_minmax_top100_periodic_knn.pt --experiment_name static_gcn_50u_seed42_final32_compat --seed 42 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### Static GCN canonical Top100 minmax scaling U50 seed 2025
Reference config: `static_gcn_50u_seed123_final32`

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/static_50u_logmass_minmax_top100_periodic_knn/camels_50u_static_logmass_minmax_top100_periodic_knn.pt --experiment_name static_gcn_50u_seed2025_final32_compat --seed 2025 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### Static GCN canonical Top100 minmax scaling U100 seed 42
Reference config: `static_gcn_100u_seed123_final32`

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/static_100u_logmass_minmax_top100_periodic_knn/camels_100u_static_logmass_minmax_top100_periodic_knn.pt --experiment_name static_gcn_100u_seed42_final32_compat --seed 42 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

### Static GCN canonical Top100 minmax scaling U100 seed 2025
Reference config: `static_gcn_100u_seed123_final32`

```bash
python3 src/training/train_static_gcn.py --dataset_path data/processed/static_100u_logmass_minmax_top100_periodic_knn/camels_100u_static_logmass_minmax_top100_periodic_knn.pt --experiment_name static_gcn_100u_seed2025_final32_compat --seed 2025 --batch_size 8 --epochs 300 --patience 40 --learning_rate 0.001 --weight_decay 1e-05 --hidden_dim 32 --num_layers 2 --dropout 0.2 --graph_pooling mean --conv_type gcn --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --grad_clip_norm 1.0 --device cuda
```

## Runs That Must Not Be Repeated

- Do not repeat debug, archive, or diagnostic folders for thesis tables.
- Do not rerun the 500U Top500 minmax seed42 runs; compatible versions already exist.
- Do not rerun kNN ablations unless a new scientific question is added; k=4/6/8/12 are already complete for three seeds.
- Do not count `evolvegcn_h_750u_top1000_h32_seed42_none_mlp_relu_repeat` as an independent seed.

## Should Scaling Be Extended to Five Seeds?

Not yet. First fix the incompatible 20U/50U/100U seed42 and seed2025 rows if a clean scaling table is required. Extending the scaling study to five seeds would require many additional runs and is lower priority than the matched 750U Static GCN baseline.
