from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def load_dataset(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_temporal_summary(sample: dict) -> np.ndarray:
    """
    Works for temporal dataset samples with:
    A_seq, X_seq, mask_seq OR A, X, mask depending on your saved format.
    """

    if "X_seq" in sample:
        X = to_numpy(sample["X_seq"])
        mask = to_numpy(sample["mask_seq"])
    elif "Nodes_list" in sample:
        X = to_numpy(sample["Nodes_list"])
        mask = to_numpy(sample["mask_list"])
    elif "X" in sample:
        X = to_numpy(sample["X"])
        mask = to_numpy(sample["mask"])
    else:
        raise KeyError(f"Cannot find X_seq, Nodes_list, or X in sample keys: {sample.keys()}")

    if X.ndim == 2:
        X = X[None, :, :]       # [1, N, F]
        mask = mask[None, :, :] # [1, N, 1]

    mask = mask.squeeze(-1) > 0

    features = []

    for t in range(X.shape[0]):
        Xt = X[t]
        mt = mask[t]

        valid = Xt[mt]

        if valid.shape[0] == 0:
            features.extend([0.0] * 20)
            continue

        mass = valid[:, 0]
        pos = valid[:, 1:4]
        vel = valid[:, 4:7]

        speed = np.linalg.norm(vel, axis=1)

        summary = [
            float(valid.shape[0]),

            float(np.mean(mass)),
            float(np.std(mass)),
            float(np.min(mass)),
            float(np.max(mass)),
            float(np.median(mass)),

            float(np.mean(pos[:, 0])),
            float(np.mean(pos[:, 1])),
            float(np.mean(pos[:, 2])),
            float(np.std(pos[:, 0])),
            float(np.std(pos[:, 1])),
            float(np.std(pos[:, 2])),

            float(np.mean(vel[:, 0])),
            float(np.mean(vel[:, 1])),
            float(np.mean(vel[:, 2])),
            float(np.std(vel[:, 0])),
            float(np.std(vel[:, 1])),
            float(np.std(vel[:, 2])),

            float(np.mean(speed)),
            float(np.std(speed)),
        ]

        features.extend(summary)

    return np.asarray(features, dtype=np.float32)


def get_target(sample: dict) -> float:
    y = sample["target"]
    if torch.is_tensor(y):
        return float(y.detach().cpu().view(-1)[0])
    return float(y)


def load_split_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Split config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["train_ids", "val_ids", "test_ids"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Split config missing required key: {key}")
        if not isinstance(config[key], list):
            raise TypeError(f"Split config key {key} must be a list.")

    return config


def validate_split_ids(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    dataset_ids: list[str],
    split_config: dict | None = None,
) -> None:
    dataset_id_set = set(dataset_ids)

    split_name_to_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    for split_name, split_ids in split_name_to_ids.items():
        if len(split_ids) == 0:
            raise ValueError(f"{split_name} split is empty.")

        duplicate_count = len(split_ids) - len(set(split_ids))
        if duplicate_count > 0:
            raise ValueError(
                f"{split_name} split contains {duplicate_count} duplicate IDs."
            )

        missing_ids = sorted(set(split_ids) - dataset_id_set)
        if missing_ids:
            raise ValueError(
                f"{split_name} split contains IDs not present in dataset: "
                f"{missing_ids[:20]}"
            )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlaps = {
        "train_val": sorted(train_set & val_set),
        "train_test": sorted(train_set & test_set),
        "val_test": sorted(val_set & test_set),
    }

    nonempty_overlaps = {
        name: ids for name, ids in overlaps.items() if len(ids) > 0
    }

    if nonempty_overlaps:
        raise ValueError(f"Split IDs overlap: {nonempty_overlaps}")

    combined_ids = train_set | val_set | test_set
    if combined_ids != dataset_id_set:
        missing_from_split = sorted(dataset_id_set - combined_ids)
        extra_in_split = sorted(combined_ids - dataset_id_set)
        raise ValueError(
            "Split IDs do not exactly cover dataset IDs. "
            f"Missing from split: {missing_from_split[:20]}; "
            f"extra in split: {extra_in_split[:20]}"
        )

    if split_config is not None:
        expected_sizes = {
            "train": split_config.get("num_train_universes"),
            "val": split_config.get("num_val_universes"),
            "test": split_config.get("num_test_universes"),
        }

        actual_sizes = {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        }

        for split_name, expected_size in expected_sizes.items():
            if expected_size is None:
                continue

            if int(expected_size) != actual_sizes[split_name]:
                raise ValueError(
                    f"{split_name} split size mismatch: "
                    f"expected {expected_size}, got {actual_sizes[split_name]}"
                )


def build_arrays_for_ids(
    ids: list[str],
    features_by_id: dict[str, np.ndarray],
    targets_by_id: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    X = np.vstack([features_by_id[uid] for uid in ids])
    y = np.asarray([targets_by_id[uid] for uid in ids], dtype=np.float32)

    return X, y


def evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)

    results = {}

    for split, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = mean_squared_error(y, pred) ** 0.5

        results[split] = {
            "mae": float(mae),
            "rmse": float(rmse),
        }

    print(f"\n{name}")
    print("-" * 60)
    print(f"Train MAE={results['train']['mae']:.6f} RMSE={results['train']['rmse']:.6f}")
    print(f"Val   MAE={results['val']['mae']:.6f} RMSE={results['val']['rmse']:.6f}")
    print(f"Test  MAE={results['test']['mae']:.6f} RMSE={results['test']['rmse']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--output_root", default="experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--split_config_path", type=str, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    data = load_dataset(dataset_path)

    universe_ids = sorted(data.keys())

    X_list = []
    y_list = []

    for uid in universe_ids:
        sample = data[uid]
        X_list.append(extract_temporal_summary(sample))
        y_list.append(get_target(sample))

    X = np.vstack(X_list)
    y = np.asarray(y_list, dtype=np.float32)

    features_by_id = {
        uid: features for uid, features in zip(universe_ids, X_list)
    }
    targets_by_id = {
        uid: target for uid, target in zip(universe_ids, y_list)
    }

    print("=" * 80)
    print("SUMMARY FEATURE BASELINE")
    print("=" * 80)
    print("Dataset:", dataset_path)
    print("Universes:", len(universe_ids))
    print("Feature matrix:", X.shape)
    print("Target min:", float(y.min()))
    print("Target max:", float(y.max()))
    print("Target mean:", float(y.mean()))

    split_config = None

    if args.split_config_path is not None:
        split_config_path = Path(args.split_config_path)
        split_config = load_split_config(split_config_path)

        train_ids = split_config["train_ids"]
        val_ids = split_config["val_ids"]
        test_ids = split_config["test_ids"]

        validate_split_ids(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            dataset_ids=universe_ids,
            split_config=split_config,
        )

        X_train, y_train = build_arrays_for_ids(
            train_ids,
            features_by_id,
            targets_by_id,
        )
        X_val, y_val = build_arrays_for_ids(
            val_ids,
            features_by_id,
            targets_by_id,
        )
        X_test, y_test = build_arrays_for_ids(
            test_ids,
            features_by_id,
            targets_by_id,
        )

        split_source = str(split_config_path)

    else:
        train_ids, temp_ids, X_train, X_temp, y_train, y_temp = train_test_split(
            universe_ids,
            X,
            y,
            test_size=(1.0 - args.train_ratio),
            random_state=args.seed,
            shuffle=True,
        )

        relative_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)

        val_ids, test_ids, X_val, X_test, y_val, y_test = train_test_split(
            temp_ids,
            X_temp,
            y_temp,
            test_size=relative_test_ratio,
            random_state=args.seed,
            shuffle=True,
        )

        validate_split_ids(
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            dataset_ids=universe_ids,
        )

        split_source = "sklearn_train_test_split"

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=args.seed,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            random_state=args.seed,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
        ),
    }

    all_results = {}

    for name, model in models.items():
        all_results[name] = evaluate_model(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )

    out_dir = Path(args.output_root) / args.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset_path": str(dataset_path),
        "seed": args.seed,
        "num_universes": len(universe_ids),
        "num_features": int(X.shape[1]),
        "target_min": float(y.min()),
        "target_max": float(y.max()),
        "target_mean": float(y.mean()),
        "split_source": split_source,
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "results": all_results,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
