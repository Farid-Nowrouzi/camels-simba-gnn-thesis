# Sparse U1000 schema specification

Schema version: `camels_temporal_sparse_v1`. Storage selector: `graph_storage: sparse_edge_index`. The historical default remains `dense_adjacency`.

## Serialized temporal sample

Each ordered universe dictionary entry contains five chronologically ordered fields: `Nodes_list[t]` float32 `[TopN,7]`, `edge_index_list[t]` int64 `[2,E_t]`, optional `edge_weight_list[t]` float32 `[E_t]` or null, and `mask_list[t]` float32 `[TopN,1]`. It also contains a float32 scalar `target`, five snapshot metadata records, schema/storage identifiers, feature/source-column definitions, periodic/k/box settings, and selection provenance. Padding remains serialized in X/mask for schema continuity, but edges refer only to indices `< num_real_nodes`.

Edge convention is `[source,target]`. Builder edges are binary, symmetric, unique, lexicographically ordered, and self-loop free. Model self-loops are inserted after edge coalescing. The exact normalized weight is `(A_ij + I_ij) / sqrt((degree_i+eps)(degree_j+eps))`, with degree taken after self-loop insertion and `eps=1e-8`, matching dense `D^-1/2(A+I)D^-1/2`.

Snapshot metadata records the source path and scale factor, real/original/valid counts, selected halo keys, raw mass ranks through their ordered position, selection SHA-256, and tie policy. Dataset metadata records schema/storage/source suite, ordered universe IDs/hash, snapshots, features and columns, raw roots, Top-N/normalization/target normalization, k/periodic/box/edge policy, node/padding/edge/target statistics, selected-halo aggregate hash, builder-config/source-manifest hashes, Git commit, Python/PyTorch/PyG versions, timestamp, dataset SHA-256, and completion status.

## Selection and neighbor algorithm

Cleaning removes nonfinite required features and nonpositive raw Mvir. Selection is stable lexicographic raw-Mvir descending then authoritative `col_1` halo ID ascending; stable original-row index is the documented fallback. Distance ties use deterministic selected-node rank.

Sparse kNN computes exact minimum-image distances one source row at a time. Per source, live temporaries are one float64 displacement `[R,3]`, one float64 squared-distance `[R]`, and one key/order `[R]`; their lifetime ends before the next source. Total work is O(R²) distance arithmetic plus bounded lexicographic selection, while peak builder memory is O(R+E). There is no `[R,R]`, `[R,R,3]`, or dense adjacency allocation in the sparse branch.

## Batching and device movement

A Static batch is a disconnected graph: X float32 `[sum R_b,7]`, edge index int64 `[2,sum E_b]`, optional weights `[sum E_b]`, graph-membership vector int64 `[sum R_b]`, pointer int64 `[B+1]`, and `num_graphs=B`. An Evolve batch is an ordered five-element list of such Static batches. Padded nodes are removed only in the collated view; serialized masks are preserved. Recursive movement sends tensors to CPU/GPU without densification.

Static final-snapshot reuse maps `Nodes_list[-1]`, `edge_index_list[-1]`, optional weight, `mask_list[-1]`, target, universe ID, and `snapshots[-1]` by reference. It performs no reselection, normalization, or graph rebuilding. Serialization is standard CPU `torch.save` plus atomic metadata and JSON completion marker.
