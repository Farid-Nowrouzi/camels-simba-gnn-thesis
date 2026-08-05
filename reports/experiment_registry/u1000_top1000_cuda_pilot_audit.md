# U1000 Top1000 CUDA Pilot Audit

Run UTC: 2026-08-05T23:31:40Z–2026-08-05T23:35:17Z

Repository branch: `thesis-sparse-integrity-hardening`

Pre-pilot commit: `0411f73fc47edeac0dc07a469e3da0d4894a8abd`

## Decision

**GO FOR TRAIN700 PILOTS.** Both authorized batch-size-1 real-data CUDA forward/MSE/backward pilots passed, and every intended forward-only batch size passed. This audit validates execution infrastructure only; it is not a scientific training result.

## Preflight and immutable identities

| Gate | Result |
|---|---|
| Required branch | PASS |
| Initial tracked/staged state | PASS — clean; branch was ahead of origin by 3 existing commits |
| Active build/training/GPU processes | PASS — none found |
| Dataset validator | PASS |
| Split-manifest validator | PASS |
| Dataset SHA-256 | `6a794c38216175212ec969283fdbd214bcdc64caecde78d24b81545e681a332a` |
| Metadata SHA-256 | `d4ea0ba0c3a1abc6f49d6856be86c7fc1226090daac8924eb6b72262d22753b9` |
| Completion SHA-256 | `4eea1a4bbbfc57d0c3420a115ae436240e0dcb1588cf47588ab2ee5809edd85a` |
| Target-table SHA-256 | `9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2` |
| Source-manifest SHA-256 | `ba22c3611a70763566ffb38a20f9b5a36fb6c1a27c3ad8030c4a7e189ce87618` |
| Split-manifest SHA-256 | `b56f35e4cbf1307344beaaf5b26cf181004d23d04fd719b678f1edb9e9924571` |
| Manifest seed/counts | 42; train 700, validation 99, test 201, unused 0, population 1000 |

The strict manifest loader verified the dataset identity, seed, ordered split hashes, disjointness, exact population coverage, canonical manifest hash, and the target-table binding. No internal split was performed. The first ordered training ID, `LH_475`, supplied both pilots.

## GPU environment

| Item | Value |
|---|---|
| GPU | NVIDIA L40 |
| Driver | 590.48.01 |
| Driver-advertised CUDA | 13.1 |
| PyTorch / compiled CUDA | 2.6.0+cu124 / 12.4 |
| Total memory | 46,068 MiB |
| Used/free before pilot process | 3 / 45,457 MiB |
| Used/free after pilot process exited | 0 / 45,460 MiB |
| Other GPU processes | None |

The model parameters and batch tensors were verified on `cuda:0`. CUDA event timing synchronized each measured region.

## EvolveGCN-H batch-size-1 pilot

Configuration: 7 features, hidden dimension 32, 2 graph-convolution layers, dropout 0.2, mean graph pooling, mean temporal pooling, linear regression head, five snapshots, no feature or target normalization.

- Universe: `LH_475` (manifest training split)
- Input: five `[1000,7]` feature tensors; 1,000 real nodes each
- Edge counts: 10,126; 10,034; 9,852; 9,838; 9,942
- Output: `[1,1]`, finite
- MSE loss: `316.7449951171875`, finite
- Forward/backward: 342.089996 ms / 109.482819 ms
- Gradients: PASS; all 14 required parameter tensors present and finite
- CUDA current allocated/reserved: 43.213379 / 118.000000 MiB
- CUDA peak allocated/reserved: 93.262695 / 118.000000 MiB
- Host RSS after backward: 2,344.277344 MiB (includes the loaded 1.047 GB production dataset)
- Sparse check: PASS; five `edge_index` tensors and no `[B,N,N]` or `[N,N]` adjacency

After object deletion, garbage collection, and cache clearing, current CUDA allocated/reserved were 16.25/40 MiB inside the still-live CUDA process; after process exit, `nvidia-smi` returned to 0 MiB used.

## Static GCN batch-size-1 pilot

Configuration: 7 features, hidden dimension 32, 3 GCN layers, mean graph pooling, existing Static MLP readout, dropout 0.2, no feature or target normalization.

- Universe: `LH_475` (same manifest training record)
- Exact final-snapshot identity: PASS; `X`, `edge_index`, and `mask` are zero-copy views of temporal list index `-1`, snapshot value `1.0`
- Input: `[1000,7]`; 1,000 real nodes; 9,942 edges
- Output: `[1,1]`, finite
- MSE loss: `0.14015208184719086`, finite
- Forward/backward: 7.624800 ms / 2.081792 ms
- Gradients: PASS; all 22 required parameter tensors present and finite
- CUDA current allocated/reserved: 16.494141 / 44.000000 MiB
- CUDA peak allocated/reserved: 23.371582 / 44.000000 MiB
- Host RSS after backward: 2,353.277344 MiB
- Sparse check: PASS; one `edge_index` tensor and no dense adjacency

## Forward-only intended batch feasibility

| Model | Batch size | Result | Output | Forward ms | Peak allocated / reserved MiB |
|---|---:|---|---|---:|---:|
| EvolveGCN-H | 2 | PASS | `[2,1]` | 10.829824 | 42.246094 / 60 |
| EvolveGCN-H | 4 | PASS | `[4,1]` | 11.098112 | 55.291504 / 84 |
| Static GCN | 4 | PASS | `[4,1]` | 3.375840 | 29.562012 / 46 |
| Static GCN | 8 | PASS | `[8,1]` | 3.234784 | 42.943359 / 58 |

Largest intended safe forward-only batches: EvolveGCN-H 4; Static GCN 8. No larger batch was tested, and no production default was changed.

## Scientific-content and safety checks

Targets, outputs, losses, and gradients were finite. Output shape was one scalar per universe. Feature dimension was 7. Evolve received exactly five snapshots; Static received exactly the same record's final snapshot. Every sparse edge index had shape `[2,E]`, stayed within real-node bounds, and could not reach padding. No dense adjacency conversion occurred.

Exactly two backward calls ran: one per batch-size-1 pilot. There was no epoch loop, optimizer construction or step, parameter update, checkpoint, prediction CSV, metrics artifact under experiments, graph rebuild, split mutation, or training output.
