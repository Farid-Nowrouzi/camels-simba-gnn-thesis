# Static GCN Graph-Protocol Comparison

This report compares lightweight metadata and source implementations only. No
dataset tensor or checkpoint was loaded.

## Dataset construction mode

Every selected 20U–500U Static candidate uses **A: a separately built native
static final-snapshot dataset**. None of the selected scaling candidates uses
the newer `temporal_final_snapshot` training-time conversion.

The five native static metadata files consistently report:

- `dataset_type=static_final_snapshot_graphs`;
- `preprocessing_version=v2_logmass_minmax_top100_periodic_knn`;
- `num_nodes=100`;
- `preferred_snapshot=1.0`;
- exact preferred-snapshot match for all 20, 50, 100, 200, and 500 successful
  universes;
- `normalization=minmax`;
- `graph_mode=knn`, `k=8`;
- periodic boundary and periodic kNN enabled;
- `box_size=25.0`;
- feature order `log10_Mvir, X, Y, Z, VX, VY, VZ`;
- node selection `top_num_nodes_by_raw_Mvir_descending`;
- graph coordinates taken from raw physical XYZ before feature normalization.

The successful-universe counts are exactly 20, 50, 100, 200, and 500, with no
metadata-reported failures. All selected candidates at a given universe count
reference the same corresponding native static dataset.

## Shared implementation

Both `src.data.build_static_graphs` and
`src.data.build_temporal_sequences` call the same
`src.data.camels_graph_utils.process_snapshot` implementation. That shared
implementation:

1. cleans the halo table;
2. sorts descending by raw `Mvir`;
3. selects at most Top-N halos;
4. constructs `log10(Mvir)` plus XYZ and velocity features;
5. pads to the requested node count and creates a binary real/padded mask;
6. builds adjacency from raw physical positions;
7. applies periodic minimum-image kNN when enabled.

The Static trainer consumes saved `A`, `X`, and `mask` directly. Its masked
pooling excludes padded nodes, and its dense GCN normalization adds self-loops.

## Protocol verdicts

| Dimension | 20U–500U Static verdict | Static versus Evolve verdict | Evidence |
|---|---|---|---|
| Feature equivalence | verified identical protocol | verified identical protocol | identical metadata feature names/columns and shared `process_snapshot` |
| Normalization equivalence | verified identical protocol | verified identical protocol | all metadata says minmax; shared feature builder |
| Top-N equivalence | verified identical protocol | verified identical protocol | `num_nodes=100`, raw-Mvir descending selection |
| Periodic-kNN equivalence | verified identical protocol | verified identical protocol | periodic flags true and shared adjacency builder |
| k equivalence | verified identical protocol | verified identical protocol | `k=8` in every metadata file |
| Box-size equivalence | verified identical protocol | verified identical protocol | `box_size=25.0` |
| Snapshot equivalence | verified identical protocol | compatible with documented default | Static metadata proves exact `a=1.0`; temporal builder sorts snapshots and its fifth item is the latest, but temporal top-level metadata does not record per-universe snapshot values and `.pt` contents were intentionally not loaded |
| Node-selection equivalence | verified identical protocol | verified identical protocol | same raw-Mvir Top-N rule and shared implementation |
| Mask/padding equivalence | verified identical protocol | verified identical protocol | same padding helper, binary-mask contract, and shared processor |
| Split-procedure equivalence | verified identical protocol | verified identical protocol | deterministic reconstruction passes; all 15 Static/Evolve split lists match exactly |

“Compatible with documented default” for the cross-model snapshot comparison
is deliberately narrower than tensor equality. Static metadata explicitly
records an exact `hlist_1.00000` match for every universe. Temporal metadata
records five successful ordered snapshots and the source chooses files from
early to late, but the lightweight top-level JSON omits each sequence's actual
snapshot values. Under the no-`.pt` constraint, byte/tensor identity of the
Static graph and Evolve's fifth graph cannot be asserted.

This limitation does not affect internal Static universe scaling: all five
Static datasets explicitly use the same final-snapshot definition and the same
builder protocol.

## Native static versus training-time temporal conversion

The current Static trainer can convert a temporal dataset by selecting
`A_list[-1]`, `Nodes_list[-1]`, and `mask_list[-1]`. That capability is not
used by the historical 20U–500U candidates.

Native Static and converted temporal final-snapshot graphs should not be
declared tensor-identical without a dedicated lightweight manifest or a future
authorized tensor comparison. They are protocol-compatible because they share
the processor, features, normalization, graph settings, node selection, and
sample IDs. The evidence level for exact snapshot/tensor equality remains
“compatible with documented default,” not “verified identical tensors.”

## Static/Evolve fairness classification

The prospective recommended Static family is **mostly controlled** relative to
the completed Evolve family:

- graph/data protocol, universe counts, seeds, and exact splits match;
- h32, two layers, dropout, mean graph pooling, optimizer settings, epochs, and
  ratios match under recommended Static Protocol A;
- temporal input and model architecture intentionally differ;
- Static batch size 8 versus Evolve batch size 4 and model-specific heads are
  disclosed residual differences.

The comparison is scientifically usable as a model-family baseline, but the
claim should be “established Static baseline versus established temporal
model,” not a pure single-variable architecture ablation.
