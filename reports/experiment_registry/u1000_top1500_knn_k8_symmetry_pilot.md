# U1000 Top1500 kNN(k=8) Static GCN symmetry pilot

## Verdict

**IMPLEMENTATION PASS — FULL DIAGNOSTIC AUTHORIZED**

This is an implementation/control-gate verdict, not a full-data scientific
conclusion. No training, optimization, checkpoint write, representation change,
or model selection occurred.

## Frozen provenance

- Implementation/protocol commit: `0a6ed95a87fd872a13284b33ba3d01353997530e`
- Protocol: `reports/experiment_registry/u1000_top1500_knn_k8_symmetry_protocol.json`
- Protocol SHA-256: `cf80146fa9c8af455e6b2ab5c5110edc3c24ce620f869295cabfd7f94474b665`
- Dataset SHA-256: `ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113`
- Checkpoint seed 42: `49ea8189bec30852bfa0879012144df8b7982291892b976368970c2c0926a487`
- Checkpoint seed 123: `dce2c8cac85ab8a3f8c3e6ae3247fcaa3cf818c9ae6364b3adf5598bd3ef50eb`
- Checkpoint seed 2025: `8be14a8dc8e7bb9a6ffb2f20ab5afe9df80e98740dc9db12584426d656b3809f`
- Split manifests (42, 123, 2025): `f5556ec5c193e7cb80f2231705edbdae32d4de206889dec308a819bdde427ab7`, `18a295106ec844848053f3040d2be3cdf73a443010be31e6e2cd962362982471`, `c233c0631b1a24d963ffccc7c6389054fddf135a7aabc4dc4ab7bf5976fab3a9`
- Graph/features: periodic sparse kNN, `k=8`, `L=25`, Top1500, raw7 order, normalization `none`
- Tolerance: `1e-6 + 1e-5 * abs(baseline_prediction)`

The dataset, metadata, split manifests, experiment configs, model architecture,
and checkpoint identities passed the runner's fail-closed audit.

## Target-independent pilot selection

IDs were ranked only by lexicographic SHA-256 of
`symmetry-pilot-v1:<universe_id>` within each frozen 201-universe test split.
Omega_m was not used.

- Seed 42 (`c0203da18590be22a1a3f900ec5147681e2eb3c946947f46f7c355438cbe277e`):
  `LH_163, LH_691, LH_30, LH_276, LH_596, LH_367, LH_640, LH_529, LH_89, LH_459, LH_605, LH_545, LH_25, LH_727, LH_745, LH_274, LH_677, LH_71, LH_448, LH_565, LH_6, LH_228, LH_382, LH_736, LH_432, LH_512, LH_182, LH_220, LH_233, LH_665, LH_373, LH_131`
- Seed 123 (`d37c4d3a2ffc32f394d77f34464faf3a7483f55cd988a07384e6c501ef919015`):
  `LH_472, LH_533, LH_455, LH_567, LH_329, LH_163, LH_691, LH_465, LH_480, LH_129, LH_661, LH_140, LH_682, LH_467, LH_529, LH_268, LH_525, LH_89, LH_147, LH_459, LH_407, LH_495, LH_440, LH_681, LH_712, LH_274, LH_677, LH_39, LH_71, LH_395, LH_29, LH_197`
- Seed 2025 (`2e0afd560560aac1aa9cc28423a5ac75e8ab50f119d42666897a0a4ea8ac582b`):
  `LH_567, LH_739, LH_77, LH_661, LH_596, LH_670, LH_338, LH_467, LH_384, LH_529, LH_268, LH_362, LH_266, LH_407, LH_265, LH_339, LH_68, LH_25, LH_727, LH_457, LH_602, LH_29, LH_565, LH_197, LH_282, LH_561, LH_704, LH_28, LH_583, LH_233, LH_453, LH_156`

The parenthesized value for each seed is the SHA-256 of the newline-delimited
ordered pilot-ID list.

## Numerical floor and model state

| Seed | Repeated originals | Max abs repeat delta | RMS repeat delta | Mean abs repeat delta | Baseline vs identity max | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 5 | 0 | 0 | 0 | 0 | 0 |
| 123 | 5 | 0 | 0 | 0 | 0 | 0 |
| 2025 | 5 | 0 | 0 | 0 | 0 | 0 |

Every model remained in eval mode, every forward used inference mode, all
parameters had gradients disabled and no accumulated gradient, and no optimizer
was constructed. Pre/post checkpoint file hashes were identical for all seeds.

## Exact topology invariance

| Family | Seed/sample graphs | Transforms | Comparisons | Exact matches | Mismatches | Max edge mismatch |
|---|---:|---:|---:|---:|---:|---:|
| Translation | 96 | 4 | 384 | 384 | 0 | 0 |
| Rotation | 96 | 24 | 2,304 | 2,304 | 0 | 0 |

Thus reuse of the original canonical `edge_index` is authorized for the full
diagnostic. No edge-set tolerance was used.

## Translation results

| Seed | Transform | Mean abs delta | RMS | P95 abs | Max abs | Failure fraction |
|---:|:---:|---:|---:|---:|---:|---:|
| 42 | T1 | 0.00826418 | 0.01136695 | 0.02375443 | 0.03213739 | 1.000 |
| 42 | T2 | 0.01030182 | 0.01343080 | 0.02746873 | 0.03914659 | 1.000 |
| 42 | T3 | 0.01215714 | 0.01755526 | 0.03525475 | 0.06055048 | 1.000 |
| 42 | T4 | 0.02099210 | 0.03186694 | 0.07024073 | 0.09734994 | 1.000 |
| 123 | T1 | 0.01198470 | 0.01626970 | 0.03103571 | 0.04963644 | 1.000 |
| 123 | T2 | 0.00883252 | 0.01109267 | 0.01980750 | 0.02169710 | 1.000 |
| 123 | T3 | 0.01269791 | 0.01729654 | 0.03747792 | 0.04350215 | 1.000 |
| 123 | T4 | 0.01513959 | 0.01915196 | 0.03741523 | 0.04239964 | 1.000 |
| 2025 | T1 | 0.01456020 | 0.01717469 | 0.02846783 | 0.03500357 | 1.000 |
| 2025 | T2 | 0.00738545 | 0.00906671 | 0.01782290 | 0.01880829 | 1.000 |
| 2025 | T3 | 0.01082614 | 0.01454695 | 0.03005721 | 0.03067465 | 1.000 |
| 2025 | T4 | 0.01646683 | 0.02203769 | 0.04780164 | 0.05761307 | 1.000 |

Aggregate translation mean absolute deltas were `0.01292881` (42),
`0.01216368` (123), and `0.01230965` (2025). Across seeds the value was
`0.01246738` (RMS `0.01770464`, P95 `0.03612111`, max `0.09734994`), with
384/384 tolerance failures.

## Rotation results

Rotation aggregates including the retained identity control:

| Seed | Mean abs delta | RMS | P95 abs | Max abs | Failures | Worst nonidentity by mean abs |
|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 0.02886333 | 0.04115965 | 0.09302082 | 0.15714186 | 736/768 | R08 (0.04206969) |
| 123 | 0.01940319 | 0.02586003 | 0.05453161 | 0.10094763 | 736/768 | R04 (0.02679579) |
| 2025 | 0.02178309 | 0.02911837 | 0.05956628 | 0.10805091 | 736/768 | R04 (0.03523875) |

Across seeds, the rotation mean absolute delta was `0.02334987` (RMS
`0.03271461`, P95 `0.06789496`, max `0.15714186`). Exactly 2,208/2,304
rotation rows failed: all 2,208 nonidentity rows failed, while all 96 identity
rows passed exactly.

Per-rotation means across all seeds (96 predictions each) were:

| Rotation | Mean abs delta | RMS | P95 abs | Max abs | Failure fraction |
|:---:|---:|---:|---:|---:|---:|
| R00 | 0.02606235 | 0.03504703 | 0.06971132 | 0.13339320 | 1.000 |
| R01 | 0.02378077 | 0.03193750 | 0.06180833 | 0.10207401 | 1.000 |
| R02 | 0.02731769 | 0.03759803 | 0.08555435 | 0.11269632 | 1.000 |
| R03 | 0 | 0 | 0 | 0 | 0.000 |
| R04 | 0.03407169 | 0.04530716 | 0.08285806 | 0.15714186 | 1.000 |
| R05 | 0.02253348 | 0.03109882 | 0.07078540 | 0.09645784 | 1.000 |
| R06 | 0.02373474 | 0.03141796 | 0.05701774 | 0.11334395 | 1.000 |
| R07 | 0.02233357 | 0.03115825 | 0.05972763 | 0.13043155 | 1.000 |
| R08 | 0.02996045 | 0.04375941 | 0.10682557 | 0.15020123 | 1.000 |
| R09 | 0.01942758 | 0.02664611 | 0.06142537 | 0.07634196 | 1.000 |
| R10 | 0.02178529 | 0.02860810 | 0.05732592 | 0.07509020 | 1.000 |
| R11 | 0.02258279 | 0.03038910 | 0.06031185 | 0.10122839 | 1.000 |
| R12 | 0.02757983 | 0.03815269 | 0.07643994 | 0.13308378 | 1.000 |
| R13 | 0.02517171 | 0.03591356 | 0.08502144 | 0.11937943 | 1.000 |
| R14 | 0.02670340 | 0.03591575 | 0.07769193 | 0.12946030 | 1.000 |
| R15 | 0.02082798 | 0.02669051 | 0.05224966 | 0.07152104 | 1.000 |
| R16 | 0.02213488 | 0.03049618 | 0.05598571 | 0.12662661 | 1.000 |
| R17 | 0.02290839 | 0.03027139 | 0.06283637 | 0.11843051 | 1.000 |
| R18 | 0.02563098 | 0.03398069 | 0.06398277 | 0.11921245 | 1.000 |
| R19 | 0.02552464 | 0.03411820 | 0.07234375 | 0.10024905 | 1.000 |
| R20 | 0.02292565 | 0.02958809 | 0.06157450 | 0.08427081 | 1.000 |
| R21 | 0.02656999 | 0.03637592 | 0.06983220 | 0.12998167 | 1.000 |
| R22 | 0.01860827 | 0.02561333 | 0.04988969 | 0.10553873 | 1.000 |
| R23 | 0.02222077 | 0.03019623 | 0.05799161 | 0.11721426 | 1.000 |

## Controlled interpretation

The nonidentity translation and rotation changes are clearly above the exactly
zero numerical floor. Because mathematical distance tests, exact pilot graph
rebuilds, identity inference, and weight integrity all passed, this is a valid
pilot observation rather than an implementation-control failure. It does not
establish the final full-test-set symmetry conclusion and does not authorize
redesign of transformations, features, graph representation, or model.

## Runtime and artifacts

- Device: CPU (CUDA driver unavailable)
- Runtime: `705.1313244019984` seconds (`11.7522` minutes)
- Prediction rows: 2,688 transformed rows; 29 core conditions per universe
- Machine summary: `reports/symmetry/u1000_top1500_knn_k8_static_gcn/pilot/symmetry_pilot_summary.json`
- Generated prediction detail: `reports/symmetry/u1000_top1500_knn_k8_static_gcn/pilot/symmetry_pilot_predictions.csv`

The full workload remains `201 × 3 × 29 = 17,487` core forwards. It was not run.
