# Sparse U1000 production-readiness decision

## Decision: GO

The optional `sparse_edge_index` path satisfies the refactor gate: it stores no dense adjacency, creates no N×N distance or N×N×3 displacement tensor, deterministically selects raw-Mvir Top-N, produces exact periodic symmetric kNN edges, preserves `D^-1/2(A+I)D^-1/2`, supports Static and Evolve with unchanged parameters/architecture, batches variable nodes without adjacency padding, reuses the identical final temporal graph, validates immutable manifests in exact order, and publishes through lock/temp/validation/checksum/fsync/atomic rename/completion marker. Dense legacy defaults, loaders, positional model calls, configs, and internal split behavior remain available.

All 18 lightweight tests passed with no failures or skips. Dense/sparse model forward and gradients met the documented tolerances. The two-universe, five-snapshot, Top16 `/tmp` smoke build and one backward pass per model passed, then all smoke files were removed.

GO means the code is ready for the separately controlled production-build review; it does not itself authorize building U1000 or running experiments. Before a future production command, review the source-manifest hashing policy and record a bounded runtime/RSS pilot because exact kNN uses O(R²) arithmetic even though its peak memory and stored form are sparse.

No production dataset, production manifest/config, scientific training, checkpoint, prediction, or experiment artifact was created or changed.
