# U1000 Top1500 integrity tamper-negative tests

Date: 2026-08-06

Test module: `tests.test_u1000_top1500_integrity_gates`

Result: **8 passed, 0 failed** (full lightweight suite: **46 passed, 0 failed**)

All fixtures are bounded temporary files. No full dataset is constructed, no CUDA API is intentionally used by these tests, no trainer is invoked, and no experiment artifact is written.

| Evidence | Mutation | Expected gate | Result |
|---|---|---|---|
| Fake bound dataset hash | Replace scoped registry dataset hash with a different valid 64-character SHA-256 | Artifact-binding preflight fails and trainer sentinel remains false | PASS |
| Pending dataset hash | Restore `PENDING_POST_BUILD` in a post-build binding | Artifact-binding preflight fails | PASS |
| Stale CUDA pilot | Keep `status=PASS` but replace dataset SHA-256 with an old valid-looking value | Pilot identity gate raises `STALE CUDA PILOT: dataset SHA...` | PASS |
| Wrong ordered partition | Keep passing-looking pilot evidence but replace ordered-partition identity | Pilot partition gate fails | PASS |
| Missing builder provenance | Validate Top1500-style metadata with all builder fields absent | Strong Top1500 provenance requirement fails | PASS |
| Wrong builder source hash | Supply the correct canonical builder path with a different valid-looking SHA-256 | Current production builder hash comparison fails | PASS |
| Correct artifact and pilot evidence | Use current temporary artifact hashes, exact seed42 manifest, and exact partition identity | Artifact and pilot gates pass | PASS |
| Top1000 backward compatibility | Validate historical schema behavior with no new builder fields | Accepted; full real Top1000 validator separately passes | PASS |

Additional full-suite coverage confirms source-manifest content tampering, same-size/restored-mtime tampering, split seed mismatches, canonical binding drift, atomic output safeguards, sparse graph invariants, and trainer manifest ordering remain protected.
