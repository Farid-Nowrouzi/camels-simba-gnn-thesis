# Controlled Top-N Halo-Count Scaling

## Motivation for Halo-Count Scaling

Top-N denotes the maximum number of raw-Mvir-ranked halos retained per
universe and snapshot; U500 denotes 500 independent simulated universes.
Increasing Top-N may expose additional mass-function, spatial, and velocity
information, while increasing graph cost.

## Top-N Selection Procedure

Invalid and nonpositive raw Mvir rows are removed, the catalogue is sorted by
raw Mvir in descending order, and the first N rows are retained independently
for every universe and snapshot. log10(Mvir) and the remaining node features
are created afterward. Raw XYZ is retained separately for periodic kNN.

## Experimental Design

EvolveGCN-H and Static GCN are analyzed separately at Top100, Top200, and
Top500 with matched seeds 42, 123, and 2025. All experiments use U500,
periodic kNN k=8, h32, mean graph pooling, raw Omega_m targets, and the
model-specific established architecture and batch size.

## Controlled Variables

Within a model, Top-N is the only intended scientific factor. Evolve uses five
snapshots, L2, temporal mean, and batch 4. Static uses a final snapshot, L3,
and batch 8. Model differences are therefore descriptive, not a causal test of
temporal processing.

## Normalization Interaction

Minmax statistics are calculated independently per universe, snapshot, and
feature after selection. Changing N changes both node population and local
minima/maxima. This is neither target leakage nor train/test leakage; it is an
inseparable part of the historical intervention.

## Verification Procedure

All 18 configs, metrics, logs, prediction CSVs, and checkpoint paths were
verified. Checkpoints were not loaded. Ordered splits contain 350/75/75 IDs,
are disjoint, cover LH_0 through LH_499, and match across all model × Top-N
rows for each seed. Metrics were independently recomputed at tolerance 1e-6.

## Quantitative Results



Top200−Top100: mean ΔMAE -0.004788 (SD 0.005264; median -0.003627; range -0.010536 to -0.000201; larger Top-N wins 3/3).
Top500−Top200: mean ΔMAE -0.006008 (SD 0.004694; median -0.005444; range -0.010958 to -0.001620; larger Top-N wins 3/3).
Top500−Top100: mean ΔMAE -0.010796 (SD 0.005376; median -0.011159; range -0.015980 to -0.005248; larger Top-N wins 3/3).

Mean MAE decreases from 0.096962
to 0.092174 and
0.086166. All three seeds improve
at both steps. The Top500−Top100 change (-0.010796)
is slightly larger than the Top100 between-seed SD
(0.009492).

## Static GCN

Top200−Top100: mean ΔMAE +0.000722 (SD 0.001511; median +0.000691; range -0.000773 to +0.002249; larger Top-N wins 1/3).
Top500−Top200: mean ΔMAE -0.000945 (SD 0.001154; median -0.000298; range -0.002277 to -0.000261; larger Top-N wins 3/3).
Top500−Top100: mean ΔMAE -0.000223 (SD 0.000733; median -0.000029; range -0.001034 to +0.000393; larger Top-N wins 2/3).

Static mean MAE is effectively flat:
0.096262,
0.096985, and
0.096039. Top200 versus Top100 and
Top500 versus Top100 are directionally mixed and much smaller than the
between-seed SD.



## Paired-Seed Analysis

Evolve improvements occur for all three seeds at every step. Static changes
are small; the Top200−Top100 and Top500−Top100 comparisons have mixed signs.
No p-values are reported because three seeds support descriptive rather than
fragile inferential evidence.

## Prediction-Compression Diagnostics



Evolve mean prediction-SD ratio rises from
0.063 to
0.394 and
0.494. Dispersion
improves in every matched seed, but remains below one; compression is reduced,
not eliminated. Evolve has no exact repeated predictions in these rows.

Static ratios remain
0.043,
0.071, and
0.064; mean exact-repeat
fractions are 0.449,
0.467, and
0.462.
Repetition does not decrease monotonically.



## Computational Scaling

k=8 yields 800, 1,600, and 4,000 directed neighbor selections per snapshot.
Dense adjacency capacity is 10,000, 40,000, and 250,000 entries. Self-loops
are absent from preprocessing and added by model layers; reciprocal neighbor
relations are symmetrized; padded nodes are excluded. Time and memory were not
recorded.

## Interpretation

Additional halos improve Evolve error and dispersion, whereas Static remains
nearly unchanged. The contrast may reflect multiple protocol differences and
cannot be attributed solely to temporal input.

## Limitations

Only three seeds are available. Halo-set nesting is expected but not
byte-proven because selected IDs were not stored and mass ties lack a stable
secondary key. kNN topology is rebuilt and is not nested. Minmax values change
with N. U750 Top1000 changes both universe count and normalization and is
excluded from the controlled trend.

## Conclusion

Additional halos provide useful information under the Evolve protocol, but
node count alone does not eliminate compression. Static shows no practically
meaningful benefit from the fivefold node increase.
