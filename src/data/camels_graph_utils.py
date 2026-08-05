from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================
# CAMELS-SIMBA Rockstar hlist column mapping
# ============================================================
# Header confirmed from raw file:
#
# scale(0), id(1), desc_scale(2), desc_id(3), num_prog(4),
# pid(5), upid(6), desc_pid(7), phantom(8), sam_Mvir(9),
# Mvir(10), Rvir(11), rs(12), vrms(13), mmp?(14),
# scale_of_last_MM(15), vmax(16),
# X/Y/Z(17,18,19), VX/VY/VZ(20,21,22), ...
#
# Legacy compatibility label used by historical static/single-snapshot tools:
#   v2_logmass_minmax_top100_periodic_knn
#
# Scientific preprocessing rule:
#   1. Select top halos using RAW Mvir.
#   2. Use log10(Mvir) as the mass feature.
#   3. Build graph edges from RAW physical X/Y/Z positions.
#   4. Use periodic boundary-aware distances for kNN/radius graphs.
#   5. Normalize node features separately for ML stability.
#
# Final feature order:
#   [log10_Mvir, X, Y, Z, VX, VY, VZ]
# ============================================================

PREPROCESSING_VERSION = "v2_logmass_minmax_top100_periodic_knn"
SPARSE_SCHEMA_VERSION = "camels_temporal_sparse_v1"
GRAPH_STORAGE_DENSE = "dense_adjacency"
GRAPH_STORAGE_SPARSE = "sparse_edge_index"
HALO_ID_COLUMN = "col_1"

DEFAULT_BOX_SIZE = 25.0


def preprocessing_version_for_config(
    *,
    num_nodes: int,
    normalization: str,
    graph_mode: str,
    k: int,
    radius: Optional[float],
    periodic_boundary: bool,
    box_size: float,
    graph_storage: str,
) -> str:
    """Return a provenance label derived entirely from effective build settings."""
    boundary = "periodic" if periodic_boundary else "nonperiodic"
    box = f"{float(box_size):g}"
    if graph_mode == "knn":
        graph = f"knn_k{int(k)}"
    elif graph_mode == "radius":
        if radius is None:
            raise ValueError("radius is required when graph_mode='radius'")
        graph = f"radius_r{float(radius):g}"
    else:
        raise ValueError(f"Unknown graph_mode: {graph_mode}")
    return (
        f"v3_logmass_{normalization}_top{int(num_nodes)}_{boundary}_"
        f"{graph}_box{box}_{graph_storage}"
    )

MASS_COLUMN = "col_10"

POSITION_COLUMNS = ["col_17", "col_18", "col_19"]

VELOCITY_COLUMNS = ["col_20", "col_21", "col_22"]

RAW_FEATURE_COLUMNS = [
    "col_10",  # Mvir: raw halo mass, used only before log transform
    "col_17",  # X position
    "col_18",  # Y position
    "col_19",  # Z position
    "col_20",  # VX velocity
    "col_21",  # VY velocity
    "col_22",  # VZ velocity
]

# Keep FEATURE_COLUMNS as a compatibility name for the rest of the project.
# Important: the first output feature is now log10(Mvir), not raw Mvir.
FEATURE_COLUMNS = RAW_FEATURE_COLUMNS

FEATURE_NAMES = [
    "log10_Mvir",
    "X",
    "Y",
    "Z",
    "VX",
    "VY",
    "VZ",
]


def read_hlist_file(path: str | Path) -> pd.DataFrame:
    """
    Read one CAMELS-SIMBA Rockstar hlist .list file.

    The raw file has comment/header lines starting with '#'.
    Data rows are whitespace-separated and have no normal CSV header.

    We create column names:
        col_0, col_1, ..., col_84
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {path}")

    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        header=None,
        engine="python",
    )

    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """
    Confirm that all columns needed for the thesis graph pipeline exist.

    Required:
        Mvir       -> col_10
        X/Y/Z      -> col_17, col_18, col_19
        VX/VY/VZ   -> col_20, col_21, col_22
    """
    required_columns = [
        MASS_COLUMN,
        *POSITION_COLUMNS,
        *VELOCITY_COLUMNS,
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required CAMELS hlist columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def clean_halo_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean invalid numerical values.

    We remove:
        - NaN
        - +inf / -inf
        - non-positive halo mass

    This keeps graph-building stable and ensures log10(Mvir) is valid.
    """
    check_required_columns(df)

    required_columns = [
        MASS_COLUMN,
        *POSITION_COLUMNS,
        *VELOCITY_COLUMNS,
    ]

    df = df.copy()
    if "_original_row_index" not in df.columns:
        df["_original_row_index"] = np.arange(len(df), dtype=np.int64)

    # Replace infinities only in required columns, then remove invalid rows.
    df[required_columns] = df[required_columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    # log10(Mvir) requires strictly positive raw mass.
    df = df[df[MASS_COLUMN] > 0]

    return df.reset_index(drop=True)


def select_top_halos(
    df: pd.DataFrame,
    num_nodes: int = 100,
    mass_column: str = MASS_COLUMN,
) -> pd.DataFrame:
    """
    Select the most massive halos using RAW Mvir.

    Important:
        Selection is done using raw Mvir, not log10(Mvir).

    This makes every graph have a consistent maximum number of nodes.
    If fewer than num_nodes halos exist, padding is handled later.
    """
    if mass_column not in df.columns:
        raise ValueError(
            f"Mass column {mass_column} not found. "
            f"Available columns: {list(df.columns)}"
        )

    tie_column = HALO_ID_COLUMN if HALO_ID_COLUMN in df.columns else "_original_row_index"
    if tie_column not in df.columns:
        working = df.copy()
        working["_original_row_index"] = np.arange(len(working), dtype=np.int64)
        tie_column = "_original_row_index"
    else:
        working = df

    selected = working.sort_values(
        by=[mass_column, tie_column],
        ascending=[False, True],
        kind="mergesort",
    ).head(num_nodes).reset_index(drop=True)
    selected.attrs["selection_tie_column"] = tie_column
    selected.attrs["selection_method"] = "raw_Mvir_desc_stable_then_tie_key_asc"

    return selected


def selection_provenance(df_selected: pd.DataFrame) -> Dict[str, object]:
    """Return deterministic selected-key/rank hashes without changing feature tensors."""
    tie_column = str(df_selected.attrs.get("selection_tie_column", "_original_row_index"))
    if tie_column not in df_selected.columns:
        tie_column = HALO_ID_COLUMN if HALO_ID_COLUMN in df_selected.columns else "_original_row_index"
    keys = [str(value) for value in df_selected[tie_column].tolist()]
    ranks = list(range(1, len(keys) + 1))
    payload = "".join(f"{rank}\t{key}\n" for rank, key in zip(ranks, keys))
    return {
        "selection_method": "raw_Mvir_desc_stable_then_tie_key_asc",
        "tie_breaking_policy": (
            "authoritative_halo_id_ascending" if tie_column == HALO_ID_COLUMN
            else "stable_original_row_index_ascending"
        ),
        "selection_key_column": tie_column,
        "selected_halo_keys": keys,
        "raw_mass_rank": ranks,
        "selection_hash_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def build_node_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Build the node feature matrix X.

    Official thesis feature set:
        [log10(Mvir), X, Y, Z, VX, VY, VZ]

    Column mapping:
        raw Mvir -> col_10 -> transformed to log10(Mvir)
        X        -> col_17
        Y        -> col_18
        Z        -> col_19
        VX       -> col_20
        VY       -> col_21
        VZ       -> col_22

    Important:
        The mass feature is log10(Mvir).
        Raw Mvir is used for selection only.
    """
    if feature_columns is None:
        feature_columns = RAW_FEATURE_COLUMNS

    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    raw_mvir = df[MASS_COLUMN].to_numpy(dtype=np.float64)

    if np.any(raw_mvir <= 0):
        raise ValueError(
            "Non-positive Mvir found while building features. "
            "Run clean_halo_dataframe before build_node_features."
        )

    log10_mvir = np.log10(raw_mvir).reshape(-1, 1).astype(np.float32)

    positions = df[POSITION_COLUMNS].to_numpy(dtype=np.float32)
    velocities = df[VELOCITY_COLUMNS].to_numpy(dtype=np.float32)

    X = np.concatenate(
        [
            log10_mvir,
            positions,
            velocities,
        ],
        axis=1,
    ).astype(np.float32)

    return X


def build_positions(df: pd.DataFrame) -> np.ndarray:
    """
    Extract raw physical 3D positions used for edge construction.

    Position columns:
        X -> col_17
        Y -> col_18
        Z -> col_19

    Important:
        These positions are NOT normalized before graph construction.
        This keeps kNN edges physically meaningful.
    """
    missing = [col for col in POSITION_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing position columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    positions = df[POSITION_COLUMNS].to_numpy(dtype=np.float32)

    return positions


def normalize_features(
    X: np.ndarray,
    normalization: str = "none",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize node features.

    Options:
        none   -> no normalization
        minmax -> each feature scaled to [0, 1] inside this snapshot
        zscore -> each feature standardized inside this snapshot

    Important:
        This function normalizes the ML feature matrix only.
        It does not affect graph edge construction.
    """
    normalization = normalization.lower()

    if normalization == "none":
        return X.astype(np.float32)

    if normalization == "minmax":
        x_min = X.min(axis=0, keepdims=True)
        x_max = X.max(axis=0, keepdims=True)

        denom = x_max - x_min
        denom = np.where(np.abs(denom) < eps, 1.0, denom)

        X_norm = (X - x_min) / denom

        return X_norm.astype(np.float32)

    if normalization == "zscore":
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)

        std = np.where(np.abs(std) < eps, 1.0, std)

        X_norm = (X - mean) / std

        return X_norm.astype(np.float32)

    raise ValueError(
        f"Unknown normalization: {normalization}. "
        "Use one of: none, minmax, zscore."
    )


def pad_nodes(
    X: np.ndarray,
    positions: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad node features and positions to fixed num_nodes.

    Returns:
        X_padded:        [num_nodes, num_features]
        pos_padded:      [num_nodes, 3]
        mask:            [num_nodes, 1]

    mask = 1 for real halo nodes
    mask = 0 for padded nodes
    """
    current_nodes = X.shape[0]
    num_features = X.shape[1]

    X_padded = np.zeros((num_nodes, num_features), dtype=np.float32)
    pos_padded = np.zeros((num_nodes, 3), dtype=np.float32)
    mask = np.zeros((num_nodes, 1), dtype=np.float32)

    keep_nodes = min(current_nodes, num_nodes)

    X_padded[:keep_nodes] = X[:keep_nodes]
    pos_padded[:keep_nodes] = positions[:keep_nodes]
    mask[:keep_nodes] = 1.0

    return X_padded, pos_padded, mask


def compute_pairwise_distances(
    positions: np.ndarray,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
) -> np.ndarray:
    """
    Compute pairwise distances between positions.

    If periodic_boundary=True, use the minimum-image convention:

        dx = min(|dx|, box_size - |dx|)

    This is the physically correct distance rule for periodic simulation boxes.

    Parameters
    ----------
    positions:
        Array with shape [N, 3].
    periodic_boundary:
        Whether to use periodic boundary-aware distances.
    box_size:
        Simulation box size. For CAMELS, the box is 25 h^-1 Mpc.

    Returns
    -------
    distances:
        Pairwise distance matrix with shape [N, N].
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"positions must have shape [N, 3], got {positions.shape}"
        )

    if periodic_boundary:
        if box_size is None or box_size <= 0:
            raise ValueError(
                "box_size must be a positive number when periodic_boundary=True."
            )

    diff = positions[:, None, :] - positions[None, :, :]

    if periodic_boundary:
        abs_diff = np.abs(diff)
        wrapped_diff = box_size - abs_diff
        diff = np.minimum(abs_diff, wrapped_diff)

    distances = np.sqrt(np.sum(diff * diff, axis=-1))

    return distances.astype(np.float32)


def build_knn_adjacency(
    positions: np.ndarray,
    mask: np.ndarray,
    k: int = 8,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
) -> np.ndarray:
    """
    Build an undirected k-nearest-neighbor adjacency matrix from raw 3D positions.

    Only valid nodes are connected.
    Padded nodes remain disconnected.

    Important:
        kNN is built from raw physical X/Y/Z positions.
        Feature normalization does not affect edges.

        If periodic_boundary=True, distances are computed with the
        minimum-image convention for a periodic simulation box.
    """
    num_nodes = positions.shape[0]
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    valid_indices = np.where(mask.reshape(-1) > 0)[0]

    if len(valid_indices) <= 1:
        return adjacency

    valid_positions = positions[valid_indices]

    distances = compute_pairwise_distances(
        positions=valid_positions,
        periodic_boundary=periodic_boundary,
        box_size=box_size,
    )

    # Do not connect a node to itself.
    np.fill_diagonal(distances, np.inf)

    effective_k = min(k, len(valid_indices) - 1)

    for local_i, global_i in enumerate(valid_indices):
        neighbor_local_indices = np.argsort(distances[local_i])[:effective_k]

        for local_j in neighbor_local_indices:
            global_j = valid_indices[local_j]
            adjacency[global_i, global_j] = 1.0
            adjacency[global_j, global_i] = 1.0

    return adjacency


def build_sparse_knn_edge_index(
    positions: np.ndarray,
    mask: np.ndarray,
    k: int = 8,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
    tie_keys: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build exact deterministic symmetric kNN edges with O(R+E) peak storage.

    For each of R real nodes, only one `[R,3]` displacement row and `[R]`
    squared-distance row are live. Total distance work is O(R^2); no `[R,R]`
    distance matrix or `[R,R,3]` displacement cube is created.

    Edge convention is `[source, target]`. The graph is symmetrized, contains
    no duplicates or self-loops, excludes padding, and is lexicographically
    ordered by `(source, target)`.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape [N,3], got {positions.shape}")
    if k <= 0:
        raise ValueError("k must be positive.")
    if periodic_boundary and box_size <= 0:
        raise ValueError("box_size must be positive for periodic kNN.")

    valid_indices = np.flatnonzero(mask.reshape(-1) > 0).astype(np.int64)
    real_count = valid_indices.size
    if real_count <= 1:
        return np.empty((2, 0), dtype=np.int64)
    valid_positions = positions[valid_indices].astype(np.float64, copy=False)
    if tie_keys is None:
        keys = np.arange(real_count, dtype=np.int64)
    else:
        keys = np.asarray(tie_keys)[:real_count]
        if keys.shape[0] != real_count:
            raise ValueError("tie_keys length must equal the number of real nodes.")
    effective_k = min(k, real_count - 1)
    edge_pairs: set[tuple[int, int]] = set()

    for local_i in range(real_count):
        diff = valid_positions - valid_positions[local_i]
        if periodic_boundary:
            abs_diff = np.abs(diff)
            diff = np.minimum(abs_diff, box_size - abs_diff)
        distance_sq = np.einsum("ij,ij->i", diff, diff)
        distance_sq[local_i] = np.inf
        order = np.lexsort((keys, distance_sq))[:effective_k]
        global_i = int(valid_indices[local_i])
        for local_j in order:
            global_j = int(valid_indices[int(local_j)])
            edge_pairs.add((global_i, global_j))
            edge_pairs.add((global_j, global_i))

    ordered = sorted(edge_pairs)
    return np.asarray(ordered, dtype=np.int64).T if ordered else np.empty((2, 0), dtype=np.int64)


def build_radius_adjacency(
    positions: np.ndarray,
    mask: np.ndarray,
    radius: float,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
) -> np.ndarray:
    """
    Build an undirected radius-based adjacency matrix from raw 3D positions.

    Two halos are connected if their spatial distance <= radius.

    If periodic_boundary=True, distances use the minimum-image convention.
    """
    if radius is None or radius <= 0:
        raise ValueError("Radius must be a positive number for radius graph mode.")

    num_nodes = positions.shape[0]
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    valid_indices = np.where(mask.reshape(-1) > 0)[0]

    if len(valid_indices) <= 1:
        return adjacency

    valid_positions = positions[valid_indices]

    distances = compute_pairwise_distances(
        positions=valid_positions,
        periodic_boundary=periodic_boundary,
        box_size=box_size,
    )

    for local_i, global_i in enumerate(valid_indices):
        for local_j, global_j in enumerate(valid_indices):
            if local_i == local_j:
                continue

            if distances[local_i, local_j] <= radius:
                adjacency[global_i, global_j] = 1.0
                adjacency[global_j, global_i] = 1.0

    return adjacency


def build_adjacency(
    positions: np.ndarray,
    mask: np.ndarray,
    graph_mode: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
) -> np.ndarray:
    """
    Build graph adjacency from halo positions.

    graph_mode:
        knn    -> connect each halo to k nearest spatial neighbors
        radius -> connect halos within a physical radius

    Important:
        The input positions should be raw physical X/Y/Z positions.
    """
    graph_mode = graph_mode.lower()

    if graph_mode == "knn":
        return build_knn_adjacency(
            positions=positions,
            mask=mask,
            k=k,
            periodic_boundary=periodic_boundary,
            box_size=box_size,
        )

    if graph_mode == "radius":
        if radius is None:
            raise ValueError("radius must be provided when graph_mode='radius'")

        return build_radius_adjacency(
            positions=positions,
            mask=mask,
            radius=radius,
            periodic_boundary=periodic_boundary,
            box_size=box_size,
        )

    raise ValueError(
        f"Unknown graph_mode: {graph_mode}. "
        "Use one of: knn, radius."
    )


def parse_snapshot_value(path: str | Path) -> float:
    """
    Extract snapshot scale factor from filename.

    Example:
        LH_0_hlist_1.00000.list -> 1.00000
        LH_0_hlist_0.51209.list -> 0.51209
    """
    path = Path(path)
    name = path.name

    marker = "_hlist_"

    if marker not in name:
        raise ValueError(f"Could not parse snapshot value from filename: {name}")

    value_text = name.split(marker, 1)[1].replace(".list", "")

    return float(value_text)


def find_universe_files(
    raw_dir: str | Path,
    universe_id: int,
) -> List[Path]:
    """
    Find all snapshot files for one LH universe.

    Example:
        universe_id = 0
        finds:
            LH_0_hlist_0.20000.list
            LH_0_hlist_0.25000.list
            LH_0_hlist_0.51209.list
            LH_0_hlist_0.75065.list
            LH_0_hlist_1.00000.list
    """
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    pattern = f"LH_{universe_id}_hlist_*.list"
    files = sorted(raw_dir.glob(pattern), key=parse_snapshot_value)

    if len(files) == 0:
        raise FileNotFoundError(
            f"No files found for universe LH_{universe_id} using pattern: {pattern}"
        )

    return files


def choose_snapshot_files(
    files: List[Path],
    num_snapshots: int = 5,
) -> List[Path]:
    """
    Choose a fixed number of snapshots across time.

    If exactly num_snapshots files exist, return all.
    If more exist, choose evenly spaced snapshots from early to late time.
    """
    if len(files) < num_snapshots:
        raise ValueError(
            f"Need at least {num_snapshots} snapshots, but found {len(files)}."
        )

    if len(files) == num_snapshots:
        return files

    indices = np.linspace(0, len(files) - 1, num_snapshots)
    indices = np.round(indices).astype(int)

    chosen = [files[i] for i in indices]

    return chosen


def process_snapshot(
    path: str | Path,
    num_nodes: int = 100,
    normalization: str = "none",
    graph_mode: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
    device: str = "cpu",
    graph_storage: str = GRAPH_STORAGE_DENSE,
) -> Dict[str, object]:
    """
    Process one snapshot file into graph tensors.

    Official preprocessing:
        v2_logmass_minmax_top100_periodic_knn

    Processing order:
        1. Read raw hlist file.
        2. Confirm required columns exist.
        3. Remove invalid rows:
              NaN, Inf, Mvir <= 0
        4. Sort halos by raw Mvir descending.
        5. Select top num_nodes halos.
        6. Build node features:
              [log10(Mvir), X, Y, Z, VX, VY, VZ]
        7. Extract raw X/Y/Z positions for graph construction.
        8. Normalize node features if requested.
        9. Pad nodes if needed.
       10. Build adjacency from raw physical positions using
           periodic boundary-aware distances if enabled.

    Returns:
        {
            "A": adjacency tensor [num_nodes, num_nodes],
            "X": node feature tensor [num_nodes, 7],
            "mask": valid node mask [num_nodes, 1],
            "path": original file path,
            "snapshot_value": scale factor
        }
    """
    path = Path(path)

    df_raw = read_hlist_file(path)
    original_num_halos = len(df_raw)

    df_clean = clean_halo_dataframe(df_raw)
    valid_num_halos = len(df_clean)

    df_selected = select_top_halos(
        df_clean,
        num_nodes=num_nodes,
        mass_column=MASS_COLUMN,
    )
    selected_num_halos = len(df_selected)

    # X contains log10(Mvir), not raw Mvir.
    X = build_node_features(df_selected)

    # positions are raw physical positions and are used for graph construction.
    positions = build_positions(df_selected)

    # Normalize ML features only.
    X = normalize_features(X, normalization=normalization)

    X_padded, positions_padded, mask = pad_nodes(
        X=X,
        positions=positions,
        num_nodes=num_nodes,
    )

    if graph_storage not in {GRAPH_STORAGE_DENSE, GRAPH_STORAGE_SPARSE}:
        raise ValueError(f"Unknown graph_storage: {graph_storage}")
    if graph_storage == GRAPH_STORAGE_SPARSE and graph_mode != "knn":
        raise ValueError("The sparse path currently supports graph_mode='knn' only.")

    selection = selection_provenance(df_selected)
    adjacency = None
    edge_index = None
    if graph_storage == GRAPH_STORAGE_DENSE:
        adjacency = build_adjacency(
            positions=positions_padded, mask=mask, graph_mode=graph_mode, k=k,
            radius=radius, periodic_boundary=periodic_boundary, box_size=box_size,
        )
    else:
        edge_index = build_sparse_knn_edge_index(
            positions=positions_padded,
            mask=mask,
            k=k,
            periodic_boundary=periodic_boundary,
            box_size=box_size,
            tie_keys=np.arange(selected_num_halos, dtype=np.int64),
        )

    if X_padded.shape != (num_nodes, 7):
        raise ValueError(
            f"Expected X shape {(num_nodes, 7)}, got {X_padded.shape}"
        )

    if adjacency is not None and adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Expected A shape {(num_nodes, num_nodes)}, got {adjacency.shape}"
        )

    if mask.shape != (num_nodes, 1):
        raise ValueError(
            f"Expected mask shape {(num_nodes, 1)}, got {mask.shape}"
        )

    if not np.isfinite(X_padded).all():
        raise ValueError(f"NaN or Inf found in X after preprocessing: {path}")

    if adjacency is not None and not np.isfinite(adjacency).all():
        raise ValueError(f"NaN or Inf found in adjacency after preprocessing: {path}")

    X_tensor = torch.tensor(X_padded, dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)

    result = {
        "X": X_tensor,
        "mask": mask_tensor,
        "path": str(path),
        "snapshot_value": parse_snapshot_value(path),
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "feature_names": FEATURE_NAMES,
        "position_columns": POSITION_COLUMNS,
        "velocity_columns": VELOCITY_COLUMNS,
        "mass_column": MASS_COLUMN,
        "mass_feature": "log10_Mvir",
        "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "normalization": normalization,
        "graph_mode": graph_mode,
        "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "k": k,
        "radius": radius,
        "num_nodes": num_nodes,
        "original_num_halos": original_num_halos,
        "valid_num_halos": valid_num_halos,
        "selected_num_halos_before_padding": selected_num_halos,
        "preprocessing_version": PREPROCESSING_VERSION,
        "periodic_boundary": periodic_boundary,
        "periodic_boundary_knn": bool(periodic_boundary and graph_mode.lower() == "knn"),
        "box_size": box_size,
        "graph_storage": graph_storage,
        "schema_version": SPARSE_SCHEMA_VERSION if graph_storage == GRAPH_STORAGE_SPARSE else None,
        "num_real_nodes": selected_num_halos,
        **selection,
    }
    if adjacency is not None:
        result["A"] = torch.tensor(adjacency, dtype=torch.float32, device=device)
    else:
        result["edge_index"] = torch.tensor(edge_index, dtype=torch.long, device=device)
        result["edge_weight"] = None
    return result


def build_universe_sequence(
    raw_dir: str | Path,
    universe_id: int,
    target: float,
    num_snapshots: int = 5,
    num_nodes: int = 100,
    normalization: str = "none",
    graph_mode: str = "knn",
    k: int = 8,
    radius: Optional[float] = None,
    periodic_boundary: bool = True,
    box_size: float = DEFAULT_BOX_SIZE,
    device: str = "cpu",
    graph_storage: str = GRAPH_STORAGE_DENSE,
) -> Dict[str, object]:
    """
    Build a temporal graph sequence for one universe.

    One universe becomes:

        {
            "A_list":      [A_t1, A_t2, A_t3, A_t4, A_t5],
            "Nodes_list":  [X_t1, X_t2, X_t3, X_t4, X_t5],
            "mask_list":   [mask_t1, mask_t2, mask_t3, mask_t4, mask_t5],
            "target":      Omega_m,
            "snapshots":   metadata
        }
    """
    all_files = find_universe_files(raw_dir=raw_dir, universe_id=universe_id)
    chosen_files = choose_snapshot_files(
        files=all_files,
        num_snapshots=num_snapshots,
    )

    preprocessing_version = preprocessing_version_for_config(
        num_nodes=num_nodes,
        normalization=normalization,
        graph_mode=graph_mode,
        k=k,
        radius=radius,
        periodic_boundary=periodic_boundary,
        box_size=box_size,
        graph_storage=graph_storage,
    )

    A_list = []
    edge_index_list = []
    Nodes_list = []
    mask_list = []
    snapshots = []

    for snapshot_path in chosen_files:
        snapshot = process_snapshot(
            path=snapshot_path,
            num_nodes=num_nodes,
            normalization=normalization,
            graph_mode=graph_mode,
            k=k,
            radius=radius,
            periodic_boundary=periodic_boundary,
            box_size=box_size,
            device=device,
            graph_storage=graph_storage,
        )

        if graph_storage == GRAPH_STORAGE_DENSE:
            A_list.append(snapshot["A"])
        else:
            edge_index_list.append(snapshot["edge_index"])
        Nodes_list.append(snapshot["X"])
        mask_list.append(snapshot["mask"])

        snapshots.append(
            {
                "path": snapshot["path"],
                "snapshot_value": snapshot["snapshot_value"],
                "preprocessing_version": preprocessing_version,
                "feature_names": snapshot["feature_names"],
                "mass_feature": snapshot["mass_feature"],
                "node_selection": snapshot["node_selection"],
                "normalization": snapshot["normalization"],
                "graph_mode": snapshot["graph_mode"],
                "graph_positions": snapshot["graph_positions"],
                "k": snapshot["k"],
                "radius": snapshot["radius"],
                "original_num_halos": snapshot["original_num_halos"],
                "valid_num_halos": snapshot["valid_num_halos"],
                "selected_num_halos_before_padding": snapshot[
                    "selected_num_halos_before_padding"
                ],
                "periodic_boundary": snapshot["periodic_boundary"],
                "periodic_boundary_knn": snapshot["periodic_boundary_knn"],
                "box_size": snapshot["box_size"],
                "num_real_nodes": snapshot["num_real_nodes"],
                "selection_hash_sha256": snapshot["selection_hash_sha256"],
                "tie_breaking_policy": snapshot["tie_breaking_policy"],
                "selected_halo_keys": snapshot["selected_halo_keys"],
            }
        )

    target_tensor = torch.tensor(
        float(target),
        dtype=torch.float32,
        device=torch.device(device),
    )

    result = {
        "Nodes_list": Nodes_list,
        "mask_list": mask_list,
        "target": target_tensor,
        "snapshots": snapshots,
        "feature_columns": FEATURE_COLUMNS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "feature_names": FEATURE_NAMES,
        "position_columns": POSITION_COLUMNS,
        "velocity_columns": VELOCITY_COLUMNS,
        "mass_column": MASS_COLUMN,
        "mass_feature": "log10_Mvir",
        "node_selection": "top_num_nodes_by_raw_Mvir_descending",
        "normalization": normalization,
        "graph_mode": graph_mode,
        "graph_positions": "raw_physical_XYZ_before_feature_normalization",
        "num_nodes": num_nodes,
        "num_snapshots": num_snapshots,
        "preprocessing_version": preprocessing_version,
        "periodic_boundary": periodic_boundary,
        "periodic_boundary_knn": bool(periodic_boundary and graph_mode.lower() == "knn"),
        "box_size": box_size,
        "graph_storage": graph_storage,
        "schema_version": SPARSE_SCHEMA_VERSION if graph_storage == GRAPH_STORAGE_SPARSE else None,
    }
    if graph_storage == GRAPH_STORAGE_DENSE:
        result["A_list"] = A_list
    else:
        result["edge_index_list"] = edge_index_list
        result["edge_weight_list"] = None
    return result
