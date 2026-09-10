# Generated and local artifact policy

Phase 3 classified the 740 visible untracked files present after canonicalization. They comprise 712 experiment artifacts, 27 generated report artifacts, and this validation script before staging.

| Class | Observed count | Policy |
|---|---:|---|
| Experiment predictions | 351 | Keep locally; do not add wholesale. Headline test predictions are registered by SHA-256 in the final artifact manifest. |
| Experiment configs | 119 | Keep visible and local. Headline configs are registered by SHA-256; do not broadly ignore configs because they are reproducibility-critical. |
| Experiment metrics | 117 | Keep visible and local. Headline metrics are registered by SHA-256; authoritative for saved run summaries and checked against predictions. |
| Train logs | 78 | Keep local; register headline run directories, but do not add the generated collection to Git. |
| `model.joblib` | 36 | Keep local; do not add model collections to ordinary Git. Headline Gradient Boosting models are hash-registered. |
| Other experiment metadata | 11 | Keep visible pending case-specific review; recovery metadata is retained with its run. |
| Checkpoints | present but already ignored | Keep locally under the existing narrow `experiments/**/checkpoints/` rule; headline checkpoints are hash-registered. |
| Generated Top1000 presentation tables/reports | 25 | Archival generated outputs; keep visible and do not stage in Phase 3. |
| Phase 3 canonical reports and validator | 3 before staging | Track explicitly. |

No new ignore rule is added. This is deliberate: a broad rule for `experiments/**/config.json`, metrics, or predictions would hide scientifically important provenance. Existing rules continue to exclude datasets, checkpoints, model binaries with `.pt`/`.pth`/`.ckpt`, notebook outputs, and bulk `outputs/`. `git status --untracked-files=no` is the tracked-state cleanliness check; ordinary status remains verbose but honest.
