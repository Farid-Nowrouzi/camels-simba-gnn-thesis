# Thesis Experiment Completion Blueprint

## Executive decision

This audit is based on `master_experiment_registry.csv`, the raw experiment
`config.json`, `metrics.json`, prediction CSVs, train logs, lightweight dataset
metadata, source text, and notebook JSON. No dataset or checkpoint was loaded
and no training was run.

The thesis already has strong controlled ablations. It does **not** yet have a
three-seed, fully controlled 20U–500U universe-scaling family, nor a complete
500U Top100/200/500 family. The strongest deadline-aware design is **Option A:
seeds 42, 123, 2025**. Three independent seeds are conventional and sufficient
for a Bachelor thesis, match the existing scaling and kNN work, and make the
uncertainty visible. Five seeds would improve precision but costs substantially
more and does not change the primary scientific questions.

## A. Historical research progression

This is a research narrative, not a single controlled scaling result.

|Stage|U|Top-N|norm|model / architecture|k|pool / head|training|seeds|mean test MAE|purpose|quality|evidence|lesson|
|---|---:|---:|---|---|---:|---|---|---|---:|---|---|---|---|
|Pilot|20|100|minmax|Evolve H, h32, L2|8|mean/mean, MLP|B4, 200–300 ep|42,123,2025|not pooled|feasibility|historical, protocol mixed|`experiments/evolvegcn_h_20u_*`|pipeline works, but very small test sets are unstable|
|Pilot|20|100|minmax|Static GCN, h32, L2|8|mean, MLP-like regressor|B4–8, 200–300 ep|42,123,2025|not pooled|static comparator|historical, protocol mixed|`experiments/static_gcn_20u_*`|no reliable architecture ordering at 20U|
|Early scale|50|100|minmax|Evolve H, h32, L2|8|mean/mean, MLP|B4, 200–300 ep|42,123,2025|not pooled|more universes|historical, protocol mixed|`experiments/evolvegcn_h_50u_*`|variance remains large|
|Early scale|50|100|minmax|Static GCN, h32, L2|8|mean, MLP-like|B4–8, 200–300 ep|42,123,2025|not pooled|static comparator|historical, protocol mixed|`experiments/static_gcn_50u_*`|static and temporal scores are close|
|Intermediate|100|100|minmax|Evolve H, h32, L2|8|mean/mean, MLP|B4, 200–300 ep|42,123,2025|not pooled|scaling|historical, protocol mixed|`experiments/evolvegcn_h_100u_*`|seed-123 outcome shows instability|
|Intermediate|100|100|minmax|Static GCN, h32, L2|8|mean, MLP-like|B4–8, 200–300 ep|42,123,2025|not pooled|scaling|historical, protocol mixed|`experiments/static_gcn_100u_*`|same protocol caveat|
|Canonical anchor|200|100|minmax|Evolve H, h32, L2|8|mean/mean, MLP|B4, 300 ep|42,123,2025|0.100152|cleaner scale point|controlled within U|`experiments/evolvegcn_h_200u_*_final32`|more universes did not monotonically solve error|
|Canonical anchor|200|100|minmax|Static GCN, h32, L2|8|mean, MLP-like|B8, 300 ep|42,123,2025|0.101222|cleaner scale point|controlled within U|`experiments/static_gcn_200u_*_final32`|near parity with Evolve|
|Large Top100|500|100|minmax|Evolve H, h32, L2|8|mean/mean, MLP|B4, 300 ep|42,123,2025|0.096962|universe scaling anchor|controlled|`experiments/evolvegcn_h_500u_seed*_final32`|modest gain, seed spread persists|
|Large Top100|500|100|minmax|Static GCN, h32, L2|8|mean, MLP-like|B4, 300 ep|42,123,2025|0.099009|universe scaling anchor|controlled|`experiments/static_gcn_500u_seed*_final32`|near parity; batch differs from lower-U canonical static protocol|
|More halos|500|200|minmax|Evolve H h32 L2 / Static h32 L3|8|mean; Evolve MLP|B4/8, 300 ep|42,123,2025|0.092174 / 0.096985|Top-N study|controlled at Top200; cross-N partly controlled|`experiments/*500u_top200*`|Top200 gives no decisive gain|
|Top500 + normalization|500|500|none|minmax/none/zscore; Evolve h32 L2, Static h32 L3|8|mean; Evolve MLP|B4/8, 300 ep|42,123,777,999,2025|0.065798 / 0.045304 (none)|normalization study|controlled|`experiments/*500u_top500*_{none,minmax,zscore}`|raw-scale features dominate normalized variants|
|Final architecture|750|1000|none|Evolve H, h32, L2|8|mean/mean or last; linear/MLP|B4, 300 ep|42,123,777,999,2025|0.055843 linear-mean; 0.055351 linear-last|head/pooling study|controlled within 750U|`experiments/evolvegcn_h_750u_top1000_*`|linear head helps; temporal-last is only marginally better|

The 750U Top1000 result is **not** a universe-scaling continuation: universe
count, Top-N, normalization, split ratios, and head changed.

## B. Clean controlled universe scaling

Intended Evolve protocol: Top100, minmax, periodic kNN k=8, same five
snapshots/features, h32, L2, dropout .2, graph mean, temporal mean, MLP, B4,
300 epochs, patience 40, LR .001, WD 1e-5, 70/15/15, seed-specific deterministic
splits. Intended Static protocol is the same except final snapshot and its
model-specific architecture. The existing Static records use L2 in the
canonical Top100 line (not the later L3 Top-N line).

|Model|U|42|123|2025|status and exact incompatibility|
|---|---:|---|---|---|---|
|Evolve H|20|incompatible/re-run|compatible/reuse|incompatible/re-run|42/2025 have 200, not 300 epochs|
|Evolve H|50|incompatible/re-run|compatible/reuse|incompatible/re-run|42/2025 have 200, not 300 epochs|
|Evolve H|100|incompatible/re-run|compatible/reuse|incompatible/re-run|42/2025 have 200, not 300 epochs|
|Evolve H|200|compatible/reuse|compatible/reuse|compatible/reuse|all audited fields match|
|Evolve H|500|compatible/reuse|compatible/reuse|compatible/reuse|all audited fields match|
|Static GCN|20|incompatible/re-run|compatible/reuse|incompatible/re-run|42/2025 B4/200 ep vs B8/300|
|Static GCN|50|incompatible/re-run|compatible/reuse|incompatible/re-run|42/2025 B4/200 ep vs B8/300|
|Static GCN|100|incompatible/re-run|compatible/reuse|incompatible/re-run|42/2025 B4/200 ep vs B8/300|
|Static GCN|200|compatible/reuse|compatible/reuse|compatible/reuse|B8/300, all audited fields match|
|Static GCN|500|incompatible with lower-U static protocol|incompatible|incompatible|B4, whereas canonical 20–200 reference is B8|

Thus seed 123 is clean across 20–500 for Evolve, but even the Static seed-123
line has a 500U batch-size mismatch. The strongest fully fixed design requires
six Evolve reruns and **nine Static reruns** (six low-U protocol replacements
plus three 500U B8 replacements), unless batch size is explicitly treated as a
benign implementation setting. The decision matrix uses the strict scientific
interpretation.

Recommendation: use three seeds, never pool the legacy 20–100 rows with the
canonical replacements, and report seed-specific split signatures from
`controlled_experiment_matrix.csv`.

## C. Clean controlled Top-N scaling

At 500U, Top100 and Top200 exist for both models. Top500 minmax exists, but:

- Evolve Top100/200 use h32 L2, B4, 300 epochs and match the Top500 minmax
  configuration; the legacy seed-42 Top500 config omits explicit default
  fields but is behaviorally compatible. Therefore Evolve has a reusable
  three-seed Top100→200→500 line.
- Static Top100 uses L2/B4, while Top200 and Top500 use L3/B8. It is not a
  controlled Top-N line. Clean it with separate Static Top100 L3/B8 runs for
  seeds 42/123/2025; the existing Top200 and minmax Top500 rows are reusable.
- Top1000 exists only at 750U with none normalization, 60/13.33/26.67 splits
  and a linear head. It is excluded.

## D. Graph-construction ablations

The kNN ablation is controlled within each model: U500, Top100, minmax,
periodic graph, seeds 42/123/2025, h64, 300 epochs, and model-specific layers
(Evolve L2/B4; Static L3/B8). Only k changes.

|Model|h|layers|k=4|k=6|k=8|k=12|
|---|---:|---:|---:|---:|---:|---:|
|Evolve H|64|2|0.097773 ± 0.011783|0.096895 ± 0.009886|0.096562 ± 0.009954|0.097453 ± 0.011609|
|Static GCN|64|3|0.097318 ± 0.010580|0.096765 ± 0.009289|0.096295 ± 0.009144|0.097433 ± 0.010385|

Answers: (1) yes; (2) yes; (3) no—different model depth, batch size and temporal
information prevent a pure absolute architecture claim; (4) no, h64 does not
invalidate the within-family k comparison because it is fixed; (5) no h32
rerun is warranted. It would mostly provide stylistic alignment, while the
scientific conclusion—k has negligible effect relative to seed variation—is
already clear.

## E. Architecture and readout ablations

|Ablation|hypothesis / changed variable|fixed design|seeds|MAE mean ± SD|result|more runs?|
|---|---|---|---|---|---|---|
|Head|linear avoids MLP collapse|750U Top1000 none, k8, h32 L2, mean/mean|5|MLP 0.061404 ± .005142; linear 0.055843 ± .008748|linear wins in mean; seed-dependent|no|
|Graph pooling|mean_max retains extremes|same 750U linear family|5|mean .055843 ± .008748; mean_max .080973 ± .010152|failed; mean is clearly better|no|
|Temporal pooling|last snapshot may retain final signal|same 750U linear/mean family|5|mean .055843 ± .008748; last .055351 ± .007362|near tie; no robust gain|no|
|Graph convolution|GraphSAGE aggregation may help|500U Top500 minmax, h32 L3, seed42|1|GCN .086305; GraphSAGE .087284|no evidence of benefit|no; one-seed exploratory only|

The single-seed 500U mean-vs-mean_max result is exploratory and should not be
mixed with the completed five-seed 750U graph-pooling ablation.

## F. Static versus temporal

**Established-model comparison:** use the matched-seed 500U Top100 h32 L2
family as descriptive evidence. Scores are very close, but static and temporal
processing differ by definition and batch behavior must be disclosed.

**Closely matched final-snapshot comparison:** at 500U Top500, compare Static
final snapshot and Evolve temporal model only within the same normalization,
seed, split, Top-N, k and training regime, while labeling depth (L3 vs L2) and
temporal access as architecture differences. With none normalization, Static
0.045304 ± .003725 beats Evolve 0.065798 ± .004922.

The missing 750U Static family can reuse
`data/processed/temporal_750u_none_top1000_periodic_knn/camels_750u_temporal_logmass_none_top1000_periodic_knn.pt`
(15.13 GB). No graph rebuild is needed:
`dataset_format=temporal_final_snapshot` selects the already saved final
`A_list[-1]`, nodes, and mask in memory. RAM risk is **very high** because the
whole temporal file is loaded before conversion and both dictionaries can
coexist. Use seeds 42/123/2025 and the exact corresponding
`evolvegcn_h_750u_top1000_h32_seed{seed}_none_linear_head_temporal_last/config.json`
split references. Static h32 L3 versus Evolve h32 L2 and lack of temporal
history remain intentional architecture confounders. A Static linear head is
not scientifically necessary: it would add a second question and current
Static uses its established regressor. Order: code review and RAM dry
inspection first, then seed 42, then 123 and 2025 only if seed 42 completes
safely.

## G. Baselines

The raw-none Top500 summary baseline reports Gradient Boosting MAE 0.012769,
Random Forest 0.013859, and Ridge 0.021594 on the exact seed-42 graph split.
Its current classification is **likely valid but needs one confirmation**:
the implementation and saved IDs show no split leakage or alignment error, but
the extraordinary raw-mass signal should be independently confirmed against
the raw catalog column semantics/target-generation provenance before calling
it validated. See `baseline_validation_audit.md`.

## H. Diagnostics and validation studies

Prediction-collapse, representation variance, head analysis, split-target
distribution, summary-feature, hybrid, and debug-overfit studies are
diagnostic evidence. They explain design decisions but must not be seed-filled
or promoted into final model tables. The duplicate 750U seed-42 MLP repeat is
not an independent seed.

## I. Missing experiments

P0 and P1 rows are enumerated in `missing_run_decision_matrix.csv`. Under the
strict blueprint there are 21 recommended new runs: 15 P0 canonical scaling
runs, 3 P0 Static Top-N runs, and 3 P1 750U Static runs. The strict scaling
replacement is Static L2/B8, whereas the Top-N reference requires L3/B8, so
these cannot be the same run.

If the supervisor accepts batch size as non-scientific for the Static scaling
plot, the three 500U Static L2/B8 reruns can be waived and the total falls to
18.

## J. Experiments not to repeat

- Five-seed extensions for universe scaling or kNN.
- Any kNN h32 reproduction.
- Completed normalization, head, graph-pooling, or temporal-pooling families.
- Debug, archive, diagnostic, duplicate, target-normalization, hybrid-summary,
  activation pilot, or single-seed GraphSAGE families.
- Existing Top200 or Top500 normalization rows.
- 750U Evolve rows.
- A Static linear-head family unless a later thesis question explicitly
  targets readout parity.

## Final recommendation

Review and commit the dirty training/model implementation before any final
run. Highest-value immediate action after that review is to complete the
three-seed canonical scaling gaps, beginning with the low-cost 20U Evolve and
Static replacements. Attempt the 15.13-GB 750U Static seed-42 run only after a
memory-safety review; it is scientifically valuable but less fundamental than
the requested clean scaling table.
