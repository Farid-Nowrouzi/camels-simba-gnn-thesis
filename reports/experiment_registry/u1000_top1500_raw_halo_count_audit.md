# U1000 Top1500 raw halo-count audit

This read-only audit inspected each raw catalogue with the production builder's validity rule: finite `Mvir`, `X/Y/Z`, and `VX/VY/VZ`, with `Mvir > 0`.

**Decision: GO.**

- Expected catalogues: 5000
- Successfully inspected: 5000
- Failed catalogues: 0
- Valid real halos: min 157, max 9795, mean 6097.398
- Population standard deviation: 1722.202
- Sample standard deviation: 1722.374
- Percentiles (1/5/10/25/50/75/90/95/99): 898.760, 2415.700, 3592.900, 5350.000, 6415.000, 7259.250, 7948.200, 8346.100, 8943.090
- Snapshots with at least 1,500 real halos: 4889
- Snapshots requiring padding: 111
- Total padded slots: 66313
- Padded-slot percentage: 0.884173%
- Universes with at least one padded snapshot: 89

## Snapshot-specific statistics

| Snapshot | Min | Max | Mean | Population SD | Sample SD | >=1500 | Padded snapshots | Padded slots | Padded % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20000 | 157 | 9426 | 4703.045 | 2208.919 | 2210.024 | 911 | 89 | 59311 | 3.954067% |
| 0.25000 | 869 | 9795 | 5919.440 | 2045.861 | 2046.885 | 978 | 22 | 7002 | 0.466800% |
| 0.51209 | 3005 | 9066 | 6963.170 | 1222.982 | 1223.594 | 1000 | 0 | 0 | 0.000000% |
| 0.75065 | 3715 | 8455 | 6430.812 | 872.367 | 872.804 | 1000 | 0 | 0 | 0.000000% |
| 1.00000 | 4256 | 8388 | 6470.522 | 729.708 | 730.073 | 1000 | 0 | 0 | 0.000000% |

## Worst affected universes

| Universe | Padded snapshots | Total padded slots | Minimum real halos |
|---|---:|---:|---:|
| LH_418 | 2 | 1974 | 157 |
| LH_522 | 2 | 1837 | 235 |
| LH_802 | 2 | 1769 | 332 |
| LH_325 | 2 | 1742 | 182 |
| LH_742 | 2 | 1706 | 346 |
| LH_322 | 2 | 1700 | 206 |
| LH_267 | 2 | 1673 | 233 |
| LH_779 | 2 | 1594 | 315 |
| LH_161 | 2 | 1588 | 350 |
| LH_500 | 2 | 1484 | 422 |

## Lowest-count snapshots

| Universe | Snapshot | Valid real halos | Padded slots |
|---|---:|---:|---:|
| LH_418 | 0.20000 | 157 | 1343 |
| LH_325 | 0.20000 | 182 | 1318 |
| LH_322 | 0.20000 | 206 | 1294 |
| LH_267 | 0.20000 | 233 | 1267 |
| LH_522 | 0.20000 | 235 | 1265 |
| LH_723 | 0.20000 | 265 | 1235 |
| LH_930 | 0.20000 | 286 | 1214 |
| LH_779 | 0.20000 | 315 | 1185 |
| LH_802 | 0.20000 | 332 | 1168 |
| LH_647 | 0.20000 | 333 | 1167 |
