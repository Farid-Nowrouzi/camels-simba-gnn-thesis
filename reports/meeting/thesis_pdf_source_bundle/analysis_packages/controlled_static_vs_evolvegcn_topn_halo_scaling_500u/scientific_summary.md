# Scientific summary

## Question and design

This model-stratified analysis asks how Top100, Top200, and Top500 affect
Omega_m regression under matched U500 protocols. All three seeds (42, 123,
2025), including poor and collapsed runs, are retained. Cross-model results
are descriptive protocol comparisons.

## EvolveGCN-H

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

## Prediction compression

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

## Computational interpretation

Moving Top100 to Top500 multiplies nodes and directed neighbor selections by
five and dense adjacency capacity by 25. Evolve gains are meaningful relative
to its observed seed variability, but the remaining compression and missing
wall-time/memory measurements prevent a complete efficiency claim. Static
shows negligible predictive benefit despite the graph-size growth.

## Conclusion

Under the tested sample-local minmax protocol, additional halos provide useful
information to EvolveGCN-H, with continued improvement from Top100 through
Top500 and no clear saturation by Top200. Increasing Top-N alone does not
eliminate prediction compression. Static GCN does not use the additional halo
population effectively under its tested readout. These are descriptive
three-seed, protocol-specific findings, not universal claims about graph size.
