# Top-N halo-selection implementation audit

## Processing order

For every universe and snapshot, the shared graph utility:

1. reads the raw halo catalogue;
2. removes NaN/Inf rows and non-positive raw Mvir;
3. sorts `col_10` raw Mvir descending;
4. takes the first configured `num_nodes`;
5. transforms Mvir to log10 and appends raw XYZ and velocity features;
6. saves raw XYZ separately for topology;
7. applies optional feature normalization to selected real nodes;
8. zero-pads features and positions and creates a real-node mask;
9. constructs periodic minimum-image kNN from raw XYZ.

Selection is independent per universe and snapshot. The configured Top-N is
constant across snapshots, but fewer-than-N snapshots are supported through
padding. Temporal validation rejects zero-real-node snapshots.

## Nesting and reproducibility

Smaller Top-N results are prefixes of larger results when the same cleaned
input table and sorting behavior are used. Stored sidecars do not contain
selected halo identifiers or rank hashes, so nesting is expected rather than
byte-proven.

`pandas.DataFrame.sort_values` is called without a stable sort kind or
secondary key. Equal raw-Mvir ties, especially at a selection boundary, are
therefore not explicitly deterministic across library implementations or
versions. Graph pooling is permutation invariant, but boundary membership can
matter scientifically.

## Padding and masks

Padding occurs after normalization. Padded feature and position rows are zero;
masks are one for real nodes and zero for padded nodes. kNN considers only
mask-positive nodes, so padding does not create edges.

## Graph topology

Topology is rebuilt for every Top-N. Raw physical XYZ, periodic box size 25,
and minimum-image distances are used. The adjacency is symmetrized, excludes
self-edges during preprocessing, and model self-loops are added separately
when configured. Adding nodes can change nearest neighbors among existing
nodes, so larger graphs are not induced supergraphs of smaller graphs.

## Normalization interaction

Minmax and zscore statistics are computed after Top-N selection, independently
per universe, snapshot, and feature. Zero ranges/standard deviations use
denominator one. Therefore a minmax Top-N comparison changes node count and
the local scaling statistics. This is an intervention coupling, not leakage.
No-normalization retains log10(Mvir) plus raw feature scales.

## Targets and snapshots

Canonical U500 sidecars use the same target CSV and LH population. Static
metadata records exact preferred final snapshot a=1.0. Temporal sidecars
record five sorted snapshots but do not persist their exact scale-factor list
at dataset level. Full graph tensors were not loaded.
