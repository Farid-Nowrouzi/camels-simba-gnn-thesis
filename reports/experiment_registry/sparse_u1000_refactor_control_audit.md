# Sparse U1000 refactor control audit

Audit date: 2026-08-05. Required branch `thesis-sparse-u1000` was active at Phase A preflight. No matching trainer process was observed; the `pgrep` command matched only its own invocation. The worktree contained many pre-existing untracked experiment and registry artifacts, which are out of scope and must remain untouched.

## Phase A inspection result

**Phase B allowed.** A bounded backward-compatible implementation is technically feasible without changing the scientific architecture. The historical dense schema and default can remain intact. An explicit `graph_storage=sparse_edge_index` path can serialize edges only, batch disjoint graphs by index offsets, and compute the same `D^-1/2 (A+I) D^-1/2` operation with `index_add_`. EvolveGCN-H can retain its GRU weight evolution, snapshot order, graph/temporal pooling, and regression head while changing only adjacency application. Tiny dense/sparse equivalence fixtures can establish the computation.

The exact sparse kNN design is row-streamed: for each real source node it allocates one `[R,3]` float displacement row and `[R]` distance/key rows, selects `(distance, deterministic node rank)` lexicographically, then canonicalizes the symmetrized edge union. It performs O(R^2) distance work and O(R^2 log k) bounded selection work in the conservative implementation, but peak working storage is O(R+E); it never materializes `[R,R]` or `[R,R,3]`. This prioritizes exact equivalence and bounded memory without adding unavailable dependencies.

## Dense locations confirmed

- `src/data/camels_graph_utils.py`: `compute_pairwise_distances` creates `[N,N,3]`; `build_knn_adjacency` and `build_radius_adjacency` create `[N,N]` arrays.
- `src/data/build_temporal_sequences.py`: validates and serializes five `A_list` dense matrices and writes directly to the final path.
- `src/training/train_evolvegcn_h.py`: stacks samples to `[B,T,N,N]`; its internal seeded ratio split is unconditional.
- `src/training/train_static_gcn.py`: stacks samples to `[B,N,N]`; final-snapshot conversion currently maps dense `A_list[-1]`.
- `src/models/evolvegcn_h.py`: builds dense normalized adjacency and uses dense batch matrix multiplication at each layer/snapshot.
- `src/models/static_gcn.py`: builds dense normalized adjacency and uses dense batch matrix multiplication at each layer.

## Environment and constraints

- Python 3.10.12; PyTorch 2.6.0+cu124.
- CUDA is unavailable in this execution environment. PyTorch Geometric, torch-sparse, SciPy, scikit-learn, and pytest are not installed; no packages will be installed.
- The standard-library `unittest` framework and pure PyTorch operations are sufficient.
- Available disk at preflight: approximately 101 GiB on the workspace and `/tmp` filesystem.
- The production U1000 dataset, production split manifests/configs, scientific training, checkpoints, and prediction artifacts are prohibited and will not be created or loaded.

## File-level plan

The precise path/responsibility/change/compatibility/test/risk matrix is recorded in `sparse_u1000_refactor_file_plan.csv`. Implementation will use minimal extensions and small shared modules rather than rewriting the existing dense pipeline.

## Phase B outcome

Phase B implemented the explicit sparse schema, deterministic Top-N and topology, pure-PyTorch sparse normalization/message passing for both models, variable-node batching, immutable split manifests, and atomic checksummed output. The historical dense default and positional model APIs remain available. All bounded tests and the isolated `/tmp` smoke build passed. The project virtual environment, discovered after the system-Python probe, contains pandas 2.3.3, PyG 2.7.0, and torch-sparse 0.6.18; the implementation deliberately avoids requiring them for sparse model operations. The expected CUDA-initialization warning was recorded, and all validation ran on CPU.

No production graph data, production split, experiment configuration, checkpoint, scientific prediction, or training epoch was created. No existing graph dataset or experiment artifact was opened for mutation.
