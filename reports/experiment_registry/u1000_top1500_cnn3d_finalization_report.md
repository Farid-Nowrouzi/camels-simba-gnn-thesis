# CNN3D final scientific closure

## 1. Scientific question

Can a compact periodic voxel-field CNN predict Omega_m from the final Top1500 halo snapshot more accurately than the canonical sparse-graph and unordered-set controls under the same universe splits?

## 2. Frozen representation

U1000, Top1500, final snapshot a=1.0, periodic 25 Mpc/h box. Cell-centered cloud-in-cell deposition on a 32³ grid produces five channels: count, deposited log-mass mark, and three deposited velocity marks. Fixed divisors are [1, 10, 100, 100, 100]; accumulation is float64 and CNN input is float32. Positions determine deposition. No graph edges are used. Target Omega_m is unnormalized.

Representation fingerprint: `1c3b66855ae1597fdf15a448818f1711c01c4ca1f801877b0e8ca8dee2358865`.

## 3. Frozen architecture

Three circular-padding 3×3×3 convolutions: 5→8→16→32 channels, ReLU; 2×2×2 average downsampling after convolution 2; global average pooling; head 32→32→16→1 with ReLU and dropout 0.2. No internal normalization. All three checkpoints strictly load into the frozen model with **20,017 trainable parameters**.

## 4. CUDA deterministic compatibility patch

The frozen implementation replaces AvgPool3d(2,2) with reshape-and-mean over non-overlapping 2³ blocks. The implementation freeze records exact CPU pooling equivalence (maximum absolute difference 0), finite CPU forward/backward, and successful strict-deterministic CUDA smoke on NVIDIA L40 with CUBLAS_WORKSPACE_CONFIG=:4096:8. This preproduction patch preserves the intended pooling and parameter count. No compatibility patch or inference was performed during closure.

## 5. Production training summary

Seeds 42, 123 and 2025 each use 700 train / 99 validation / 201 test universes. AdamW, learning rate 0.001, weight decay 1e-5, batch size 8, MSE loss, gradient clipping 1.0, maximum 300 epochs, early-stopping patience 40, ReduceLROnPlateau factor 0.5/patience 10/minimum LR 1e-6. Checkpoints were selected by strict minimum validation MSE. Best epochs: 154, 138 and 101 respectively.

Protocol, implementation, all protected source/config/tests, dataset and splits passed SHA256 verification. Checkpoints, train/validation predictions, train logs and config files match recorded pre-finalization hashes. Train/validation IDs and targets also passed checks. No training artifacts were modified.

## 6. Validation results

| Seed | Best epoch | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- | --- |
| 42 | 154 | 0.0186234620785472 | 0.00048418945800115 | 0.0220043054423708 | 0.961715772382939 |
| 123 | 138 | 0.0181966779960526 | 0.000501608569350444 | 0.0223966195965026 | 0.962972496239799 |
| 2025 | 101 | 0.0190303813327443 | 0.000566967867225462 | 0.0238110870651775 | 0.959619084609481 |

## 7. Protected test results

| Seed | MAE | MSE | RMSE | R² |
| --- | --- | --- | --- | --- |
| 42 | 0.0176958989222251 | 0.000505969103570347 | 0.0224937569910041 | 0.965483600680001 |
| 123 | 0.0203646405122766 | 0.000606715039708976 | 0.0246315862199124 | 0.953409148993549 |
| 2025 | 0.0230966451675145 | 0.000860959517408485 | 0.0293421116726197 | 0.940585629224568 |

Each run is finalized/evaluated_once, with one attempt marker and exactly 201 unique test IDs in manifest order. Targets exactly equal authoritative dataset float32 Omega_m and predictions are finite. Metrics were independently recomputed from target/prediction columns in float64 and agree with metrics.json within rtol=1e-12, atol=1e-14. No new model forward passes were used.

## 8. Three-seed aggregate

| Metric | Validation mean ± sample SD | Test mean ± sample SD |
| --- | --- | --- |
| MAE | 0.0186168404691147 ± 0.000416891110107052 | 0.0203857282006721 ± 0.00270043487598814 |
| MSE | 0.000517588631525685 ± 4.36415862311437e-05 | 0.000657881220229269 ± 0.000182942701415005 |
| RMSE | 0.022737337368017 ± 0.000950358568119029 | 0.0254891516278454 ± 0.00350379141544756 |
| R2 | 0.96143578441074 ± 0.00169414796845525 | 0.953159459632706 ± 0.0124508635938067 |

Primary result: **test MAE 0.0203857282006721 ± 0.00270043487598814**. Equal seed weights; ddof=1. These are aggregates of seed metrics, not pooled predictions. Test MAE range: 0.0176958989222251 to 0.0230966451675145. No best-seed selection.

## 9. Generalization gap

| Seed | Test MAE − validation MAE |
| --- | --- |
| 42 | -0.000927563156322104 |
| 123 | 0.00216796251622401 |
| 2025 | 0.00406626383477021 |

Mean gap ± sample SD: 0.00176888773155737 ± 0.00252071865091249. Mean test MAE is about 9.5% higher than validation. Seed 42 improves slightly; seeds 123 and 2025 worsen, and test seed variation is larger. Performance remains in the same error range, with a modest average degradation and appreciable seed sensitivity. No significance claim is made.

## 10. Matched static controls

| Model | Test MAE mean ± sample SD | Parameters |
| --- | --- | --- |
| Static GCN | 0.037886619493736 ± 0.000941551444284375 | 5281 |
| PNA | 0.0448446859841916 ± 0.00830546700552573 | 51553 |
| DeepSets | 0.0480506249377581 ± 0.000985757973008614 | 4161 |
| Set Transformer | 0.0497680051409783 ± 0.00138334214596111 | 11313 |

Paired differences below are CNN MAE − control MAE. Negative means lower CNN error. All seed-specific test ID sets and targets match. These are differences of per-seed MAEs; sample SD is over three paired differences.

| Control | Seed 42 | Seed 123 | Seed 2025 | Paired mean ± sample SD |
| --- | --- | --- | --- | --- |
| Static GCN | -0.0205001024835145 | -0.0164646633821933 | -0.0155379080134838 | -0.0175008912930639 ± 0.00263840286955031 |
| PNA | -0.0222868369660567 | -0.0197519798628726 | -0.0313380565216292 | -0.0244589577835195 ± 0.00609080246928822 |
| DeepSets | -0.031474773378218 | -0.0273015717072273 | -0.0242183451258128 | -0.027664896737086 ± 0.00364183216987926 |
| Set Transformer | -0.031416704368532 | -0.0284695300593305 | -0.0282605963930562 | -0.0293822769403062 ± 0.00176496020911626 |


## 11. Summary-baseline context

| Model | Test MAE mean ± sample SD | Parameters |
| --- | --- | --- |
| Random Forest | 0.00937576106189813 ± 0.000536435658350808 | N/A (tree ensemble) |
| Gradient Boosting | 0.00917301027257102 ± 0.000511364837835855 | N/A (tree ensemble) |

Paired differences below are CNN MAE − control MAE. Negative means lower CNN error. All seed-specific test ID sets and targets match. These are differences of per-seed MAEs; sample SD is over three paired differences.

| Control | Seed 42 | Seed 123 | Seed 2025 | Paired mean ± sample SD |
| --- | --- | --- | --- | --- |
| Random Forest | 0.00880175249134823 | 0.011085412802354 | 0.0131427361226196 | 0.011009967138774 ± 0.00217147501893443 |
| Gradient Boosting | 0.00874665078132047 | 0.0115529741507059 | 0.0133385288522768 | 0.0112127179281011 ± 0.00231477143091893 |


RF/GB use 20 engineered final-snapshot summary features and 300 trees each. Neural parameter counts are not available/comparable. CNN mean MAE exceeds RF by 0.011009967138774 and GB by 0.0112127179281011 (roughly 2.17× and 2.22× their respective MAEs). Only Top1500 results are used.

## 12. Temporal-model context

| Model | Test MAE mean ± sample SD | Parameters |
| --- | --- | --- |
| EvolveGCN-H | 0.055392260822293 ± 0.00465348363322907 | 3408097 |
| EvolveGCN-O | 0.0326371543961971 ± 0.0025116319955764 | 11527 |
| GCN-GRU | 0.0293927373577706 ± 0.000481158510009573 | 11617 |
| GCN Temporal Transformer | 0.0353428422663342 ± 0.00308931467176355 | 13825 |

Paired differences below are CNN MAE − control MAE. Negative means lower CNN error. All seed-specific test ID sets and targets match. These are differences of per-seed MAEs; sample SD is over three paired differences.

| Control | Seed 42 | Seed 123 | Seed 2025 | Paired mean ± sample SD |
| --- | --- | --- | --- | --- |
| EvolveGCN-H | -0.0425626594406455 | -0.030621080775166 | -0.0318358576490511 | -0.0350065326216209 ± 0.00657192590434638 |
| EvolveGCN-O | -0.0122515223364332 | -0.0126780737840121 | -0.0118246824661298 | -0.012251426195525 ± 0.000426695667064419 |
| GCN-GRU | -0.0114258146330492 | -0.0087435812647663 | -0.00685163157348016 | -0.00900700915709855 ± 0.00229844150598966 |
| GCN Temporal Transformer | -0.0159636407126835 | -0.0130961177064412 | -0.0158115837778618 | -0.0149571140656621 ± 0.00161346239950987 |


These four temporal models receive five snapshots; they are not input-matched to the final-snapshot CNN. CNN has lower test MAE on all three seeds than each listed temporal model. This does not establish that temporal information lacks value.

## 13. Scientific interpretation

Under this frozen benchmark, the voxel-field CNN outperforms Static GCN, PNA, DeepSets and Set Transformer in mean MAE and on every paired seed. Mean MAE is about 46.2% lower than Static GCN, 54.5% lower than PNA, 57.6% lower than DeepSets and 59.0% lower than Set Transformer. It also has lower error than the contextual temporal graph models, despite receiving only one snapshot. RF and GB remain substantially more accurate.

This supports the usefulness of periodic deposition, fixed channel scaling and local convolution together as a predictive representation/model combination. A 20,017-parameter CNN achieves lower error than the 51,553-parameter PNA and 3,408,097-parameter EvolveGCN-H, but has more parameters than several other controls. This suggests useful predictive efficiency of this representation/model pairing; it does not measure compute efficiency or isolate an architecture effect. Strong summary baselines suggest that engineered aggregate information remains highly informative for this target. No model is claimed universally superior.

## 14. Limitations and registry decision

- Three seeds; sample SD describes seed variation, not a confidence interval or significance test.
- Seed test sets may overlap; paired seed differences are descriptive.
- Representation, architecture, scaling, and capacity differ; no architecture-only causal claim.
- Engineered summary models and five-snapshot temporal models are context rather than raw-input-matched static controls.
- Parameter counts alone do not measure runtime, memory, energy, or total computational efficiency.
- Result applies to the frozen U1000/Top1500 dataset, Omega_m target, split design and training protocol.

**Registry integration deferred.** An in-memory rebuild using the unmodified builder produced 401 candidate rows versus 397 existing rows, with no missing historical paths but 39 historical rows having changed values (examples: notebook_matches). Thus the required all-values-preserved condition fails. CNN currently lacks family classification and dedicated nested grid-config parsing. The registry and builder were left byte-for-byte unchanged; the global registry remains at 397 rows. No historical freeze was rewritten.

## 15. Provenance and exact freeze hashes

- Repository: `/home/ml/thesis-camels`
- Branch: `thesis-notebook17-set-transformer`
- HEAD: `e2f58f269a6941b3a5c621f84dad1d4dcd89b5ac`
- Scientific protocol SHA256: `ce934560d21b7823924f235787dd972cdb72abd8f0ebcf452ddd85db11e5b490`
- Implementation/source freeze SHA256: `692dc965ae3f3f9fac3f0fb16a16a91cb7d7aabf2ff9762cf7341bc2af63fcc8`
- Result freeze SHA256: `f9bf7a103ad8fda5fa3627745bb46b3677ecd8e7f9adc59d002fd9750478c956`
- Dataset SHA256: `ff6f6a89517c0b67a96a8733ce5778dba1524df441cc23c27fbfa4e2f5cdb113`

The result freeze is `u1000_top1500_cnn3d_result_freeze.json`, serialized with sorted keys, two-space indentation, finite JSON numbers, and a trailing newline. Its own hash is recorded here, never inside itself. It includes exact split/checkpoint/prediction/log/marker hashes, metrics, integrity evidence, control run paths and hashes, and registry audit evidence.

Authoritative comparison artifacts:

- `reports/experiment_registry/u1000_top1500_set_transformer_final_result_freeze.json` (SHA256 `63b19e8c74abc91a7ac9cd6d8d93eeaad9de8c9e3f2cb810d33e9951928df989`)
- `reports/experiment_registry/u1000_top1500_static_gcn_vs_pna_architecture_freeze.json` (SHA256 `2c9a5437d0c840dbf7719b0293471f1a57de241dcbd19569b1d62c5a913578ec`)
- `reports/experiment_registry/u1000_top1500_temporal_five_model_validation_freeze.json` (SHA256 `fec55630d759311b5c6aa845a9a12aab3a7afd7e292d1a4dab1e6da1b07f6d18`)
- `reports/experiment_registry/u1000_top1500_temporal_five_model_postfreeze_test_evaluation.json` (SHA256 `65b939fcb177e53437dadef4ff42e9ae3c76545d48905e5edfc0e510d5959de0`)

Canonical run directories and per-run prediction/config/metrics hashes are listed in the result freeze. Prior set-model/static-control freeze hashes and PNA checkpoint hashes were checked; temporal metrics agree with the prior postfreeze report. Summary control provenance comes from the existing Top1500 run configs and predictions.

## 16. Test-set protection and closure statement

Finalization was completed externally after the prior pre-test audit. This closure only reads existing predictions and checkpoints; no test inference, training, tuning, seed selection, or finalize invocation. One persistent marker per seed and evaluated_once metadata are consistent with one protected family pass. No reruns permitted.

Protected attempt accounting: **one family finalization pass, one evaluation per seed, three seed evaluations total**, consistent with the marker protocol and prior audited absence. A marker is not an independent execution counter for unlogged external actions. This closure invoked finalize zero times. Recomputing metrics from saved predictions does not constitute another model evaluation.

**PASS — CNN3D scientifically closed.** No source, checkpoint, prediction, split, dataset, metrics, historical report, global registry, or notebook was modified. Notebook 14 retains its pre-existing metadata-only change. No staging, commit or push. Notebook 18 was not created; it can proceed after closure review. Stop here.
