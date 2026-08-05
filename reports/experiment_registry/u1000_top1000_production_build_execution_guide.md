# U1000 Top1000 sparse production-build execution guide

## Authorization boundary

This guide prepares a manual, monitored build. Nothing in the launcher creates splits, starts training, uses a GPU, writes checkpoints, or creates predictions. Do not run the production launcher until the four created workflow files have been reviewed.

Expected resources are approximately 41–75 minutes, 2.5–4.5 GiB builder peak RAM, and a 0.85–1.20 GB final dataset. The launcher requires at least 5 GiB free disk, runs one CPU builder with four numerical-library threads, and lowers CPU and I/O priority.

## Exact commands

From `/home/ml/thesis-camels`, run the shell-only preflight first:

```bash
scripts/production/run_u1000_top1000_sparse_build.sh --preflight-only
```

Only after that reports `PREFLIGHT PASS`, create the tmux session:

```bash
tmux new-session -s u1000-top1000-sparse
```

Inside tmux, enter the repository and manually launch the build:

```bash
cd /home/ml/thesis-camels
scripts/production/run_u1000_top1000_sparse_build.sh
```

Detach without stopping the build by typing `Ctrl-b`, releasing both keys, and then typing `d`.

Reattach after a disconnect or intentional detach with:

```bash
tmux attach-session -t u1000-top1000-sparse
```

Run the read-only monitor from another terminal:

```bash
cd /home/ml/thesis-camels && scripts/production/monitor_u1000_top1000_build.sh
```

After the launcher reports success, run the independent read-only validator:

```bash
cd /home/ml/thesis-camels && envs/camels-gnn/bin/python scripts/validation/validate_u1000_top1000_sparse_dataset.py
```

## Expected result and output

Successful construction ends with `SUCCESS`, builder exit status zero, and all three atomic publication files present:

- `data/processed/temporal_1000u_none_top1000_periodic_knn_sparse/camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.pt`
- the adjacent `camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.metadata.json`
- the adjacent `camels_1000u_temporal_logmass_none_top1000_periodic_knn_sparse.complete`

The timestamped log is written as `logs/dataset_builds/u1000_top1000_sparse_build_YYYYMMDDTHHMMSSZ.log`. Final acceptance requires the validator's last line to be exactly `PASS` and its shell exit status to be zero.

Failure indicators include `NO-GO`, `FAILURE`, a nonzero launcher status, a Python traceback, an existing `.lock` or UUID `.tmp` file after the process has exited, a missing completion marker, checksum disagreement, or validator `FAIL`. Do not create a split from a failed or unvalidated build.

## Operational safety responses

If free or available memory becomes dangerously low, first stop or pause unrelated workloads without touching the builder. If pressure remains unsafe, reattach to the tmux session and interrupt this builder once with `Ctrl-C`; wait for it to exit and preserve the log for diagnosis. Do not start a second builder. The atomic writer is designed to remove its own temporary and lock files on exit, but inspect any leftovers rather than deleting them blindly.

If disk approaches full, stop unrelated disk-producing work and free only known, unrelated, recoverable files. If the 5 GiB safety margin cannot be restored promptly, interrupt the builder once from its tmux terminal. Never delete raw CAMELS data, an output being written, a lock owned by a live process, or an existing graph dataset to make space. After exit, review the log and atomic artifacts before deciding whether a clean restart is safe.

An SSH or terminal disconnect does not stop a builder running inside tmux. Reconnect, use the exact reattach command above, and inspect the monitor and log. Do not launch another copy because the original terminal disappeared.

During construction, do not run another graph builder, Static or EvolveGCN training, split generation, preprocessing sweep, multiprocessing job, GPU workload, checkpoint/prediction job, or any command that modifies the raw data or existing graph datasets.

Only after validation prints `PASS` may the published SHA-256 from the completion marker be used as the immutable dataset identity for a separately reviewed split manifest. Split creation and all training remain separate, explicitly authorized steps; neither happens automatically.
