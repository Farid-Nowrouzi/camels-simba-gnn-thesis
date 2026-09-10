# Phase 3 stash reconciliation

Neither stash was applied, popped, or dropped.

| Source stash | Path / scope | Classification | Decision | Current equivalent | Test coverage |
|---|---|---|---|---|---|
| `stash@{0}` (`df904674981db8c09d3591c6a5ecc690519b98b5`) | `scripts/analysis_reporting/analyze_u1000_top1000_training_scaling.py`; validation invariants, paired seed-specific tables, figures, and presentation reports | C/D: scientifically careful reporting work, but a large historical Top1000 presentation rewrite whose generated outputs already exist in the working tree | ARCHIVAL — HUMAN REVIEW. Exact functionality is useful, but integrating a 574-line historical rewrite during the final Top1500 freeze is not low-risk and is not required to reproduce a headline result. | Partial: current analyzer plus existing generated Top1000 reports; no exact source-level equivalent established | Existing Top1000 validation outputs report PASS, but the stashed source was not independently executed in Phase 3 |
| `stash@{1}` (`c5cd35716a9ec87a0cbc6cab98456f9f92d132c7`) | pre-sparse-U1000 unrelated tracked changes | D: historical / archival only | ARCHIVAL — HUMAN REVIEW | Not investigated beyond read-only inventory because Phase 3 supplied no evidence of current scientific necessity | Not applicable |

No hunk was reconstructed from either stash. This avoids silently changing report semantics and preserves both objects for later human review.
