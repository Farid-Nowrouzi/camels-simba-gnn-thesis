# U1000 target-table production readiness

## Decision

**GO** for using `outputs/target_inspection_1000u.csv` as the authoritative U1000 `target_table` source in a future, separately authorized full-SHA-256 production source manifest.

This decision authorizes no graph build, split creation, experiment configuration, or training.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Authoritative source identified | PASS | Raw CAMELS-SIMBA catalogue header key `Omega_M`; filename-derived `LH_n` identity |
| Exact universe coverage | PASS | 1,000 rows, `LH_0..LH_999`, numeric order, no missing/unknown/duplicate IDs |
| Finite targets | PASS | 1,000/1,000 numeric and finite |
| Five-snapshot agreement | PASS | 1,000/1,000 exact; maximum spread 0 |
| Independent parameter table | N/A | No independent parameter table exists in the repository/raw directories |
| U750 prefix | PASS | 750/750 exact parsed matches; maximum difference 0; identical order/fields (LF versus historical CRLF only) |
| Deterministic regeneration | PASS | Byte-identical extraction plus LF normalization in a temporary path |
| Full file SHA-256 | PASS | `9692a97760ee0e3a97cf3293f1b73911ee0a1af028617f03fe85f88f431703c2` |
| Source-manifest target entry | PASS | Full-SHA-256 build/verify accepted target entry; temporary mutation detected |
| Builder target parser | PASS | 1,000 exact `Dict[int,float]` mappings from `--targets_csv` interface |

## Statistical identity

- Minimum: `0.1002`
- Maximum: `0.49980000000000002`
- Mean: `0.29999999999999999`
- Median: `0.30000000000000004`
- Population standard deviation: `0.1154699961028838`
- Repeated target values: 0 distinct repeated values; maximum frequency 1

Detailed evidence is in the creation audit and the three CSV validation artifacts. No raw catalogue or existing target table was modified. No U1000 or repository production graph dataset was built. The creation audit records a temporary synthetic graph fixture that was unintentionally exercised by the broad source-manifest unit suite and automatically removed.
