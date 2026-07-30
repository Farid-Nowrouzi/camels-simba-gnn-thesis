# Scientific summary



## Verified conclusion



Under the tested U500 Top500 k=8 protocol, the unnormalized node-feature representation achieved lower test error in every matched seed for both models. Minmax and zscore are sample-local transformations, so this result does not imply that all normalization strategies are harmful.



## Answers to the scientific questions



1. **Does minmax improve over none for EvolveGCN-H?** No. Mean paired MAE difference (minmax − none) is +0.019202; all 5 of 5 differences are positive, so none has lower MAE in every seed.

2. **Does zscore improve over none for EvolveGCN-H?** No. Mean paired MAE difference (zscore − none) is +0.028751; all 5 of 5 differences are positive, so none has lower MAE in every seed.

3. **Does minmax improve over none for Static GCN?** No. Mean paired MAE difference (minmax − none) is +0.050682; all 5 of 5 differences are positive, so none has lower MAE in every seed.

4. **Does zscore improve over none for Static GCN?** No. Mean paired MAE difference (zscore − none) is +0.054031; all 5 of 5 differences are positive, so none has lower MAE in every seed.

5. **Are effects consistent across all five seeds?** Yes for the none-versus-normalized comparisons: none has lower MAE in all 20 model × method × seed comparisons against minmax or zscore.

6. **Are effects larger than between-seed variability?** Yes. In both models, the mean paired MAE penalties for minmax and zscore exceed the corresponding cell-level MAE standard deviations.

7. **Do normalized variants show stronger compression?** Yes. Mean prediction-SD ratios change from 0.691 (Evolve none) to 0.483/0.254, and from 0.846 (Static none) to 0.111/0.262.

8. **Do normalized variants show more repeated predictions?** For Static GCN, yes: mean exact repeated fractions are 0.427 (minmax) and 0.056 versus 0 for none. Evolve normalized runs have zero exact repeats, whereas Evolve none has a small mean repeated fraction; compression there is visible primarily through reduced dispersion.

9. **Is the pattern similar across models?** Qualitatively yes: none has the lowest error and highest mean prediction dispersion in both. The severity and exact repetition behavior differ.

10. **Is normalization a major explanation for prediction collapse?** Sample-local normalization is a plausible major contributor under this protocol, especially for Static GCN, but it is not a universal explanation and the analysis does not test global train-fitted scaling.



## Interpretation



Per-universe, per-snapshot scaling removes absolute feature-scale differences between universes. Those differences may carry Omega_m information. The observed error increase and reduced prediction dispersion are consistent with that mechanism, but this analysis does not prove the mechanism causally.



Static-versus-Evolve differences are descriptive protocol comparisons. They must not be attributed solely to temporal processing because the architectures, depth, batch sizes, and heads also differ.
