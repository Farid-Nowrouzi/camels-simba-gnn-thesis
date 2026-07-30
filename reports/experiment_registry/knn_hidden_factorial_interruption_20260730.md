# kNN × Hidden-Dimension Factorial Interruption — 2026-07-30

## Scope and safety

This was a read-only incident inspection plus minimal operational hardening. No
training, tmux launch, checkpoint/dataset load, notebook execution, graph
rebuild, experiment-artifact edit, family-specification edit, or partial-folder
move occurred. The inspected branch was `thesis-controlled-scaling`.

The evidence directory remains unchanged at:

`experiments/evolvegcn_h_u500_top100_norm-minmax_k6_h32_l2_factorial_seed42`

## Current inactive state

Read-only checks at 2026-07-30 04:39 UTC found:

- `canonical-knn-hidden-factorial` absent. The only tmux session was the
  unrelated `canonical-scaling`.
- No matching trainer, family runner, or wrapper process. The only `pgrep`
  match was the inspection command containing the search expression.
- NVIDIA L40 visible through driver 590.48.01 / CUDA 13.1, 0 MiB allocated,
  4% GPU utilization, and no listed GPU processes.
- 102 GiB free on the repository filesystem (46% used), with 3% of inodes
  used.
- 59 GiB system RAM available at inspection time. The host has no swap.

These are current-state observations, not measurements of resource use at the
time of interruption.

## Incident timeline (UTC)

| Time | Evidence |
|---|---|
| 04:13:14 | Combined-log filename timestamp and tmux launch interval begin. |
| 04:13:15.816416 | Evolve h32/L2, k4, seed 42 started. |
| 04:14:18.092603 | Seed 42 finished with exit status 0. |
| 04:14:18.093892 | Evolve h32/L2, k4, seed 123 started. |
| 04:15:32.276266 | Seed 123 finished with exit status 0. |
| 04:15:32.277399 | Evolve h32/L2, k4, seed 2025 started. |
| 04:16:34.002263 | Seed 2025 finished with exit status 0. |
| 04:16:34.003615 | Evolve h32/L2, k6, seed 42 started. |
| 04:16:38.553583 | Interrupted run `config.json` final write. |
| 04:16:47.733563 | `checkpoints/best_model.pt` final write; checkpoint was not loaded. |
| 04:16:49.145560 | Combined log final write. |
| 04:17:29.380472 | Per-run log final write: epoch 7. |
| approximately 04:17 | Login accounting records the factorial tmux pane ending. |

The three new k4 runs (seeds 42, 123, and 2025) are complete. The k6 seed-42
run is interrupted and partial.

## Exact log endings and artifact state

The exact final physical line of the combined log is:

```text
------------------------------------------------------------------------------------------
```

It is the separator immediately below the `Split details` heading. The exact
final per-run-log line is:

```text
Epoch 007 | Train MSE: 0.01428763 | Val MSE: 0.01255987 | Best Val: 0.01255987 at epoch 7 | LR: 0.00100000
```

The partial directory contains `config.json` and
`checkpoints/best_model.pt`. It lacks `metrics.json`, `train_log.csv`, and
`predictions/test_predictions.csv`. An empty `predictions/` directory is
present. The checkpoint was neither loaded nor altered.

The 40-second difference between the combined log's final write and the
per-run log's final write demonstrates that the combined stream lagged behind
the family runner's per-run file. Python block buffering is a sufficient
explanation for that lag and could hide final combined-log messages. It does
not by itself explain why the process hierarchy ended.

## System-log and host evidence

- `uptime`, `who -b`, and `last -x` show the system boot at
  2026-07-20 10:27 UTC and continuous operation through the incident. No reboot
  or shutdown occurred in the incident interval.
- `last -x` records `tmux(18124).%3` from 04:13 through 04:17 UTC.
- Both system and kernel `journalctl` queries for 04:10–04:25 returned no
  entries and warned that this user cannot see other users' messages. This is
  insufficient to rule out kernel OOM, NVIDIA Xid, cgroup termination, or
  administrator action.
- The user journal also contained no entries.
- `coredumpctl` is unavailable.
- A narrow shell-history search found no relevant tmux-session kill, process
  kill, systemctl, reboot, or shutdown command. Shell history has no reliable
  incident timestamps and is low-confidence evidence.
- No traceback, CUDA OOM, explicit signal, segmentation fault, wrapper exit
  status, or driver Xid is present in the available experiment logs.

## Original logging and exit-propagation audit

The generic family runner:

- executes one experiment at a time in a sequential loop;
- logs the exact command and experiment name before starting;
- records start UTC, finish UTC, and subprocess exit status on normal child
  termination;
- merges child stderr into stdout and writes the stream to both console and
  the per-run log;
- raises an error after a nonzero child status and stops the family;
- verifies successful runs are complete before continuing;
- skips compatible completed rows and rejects partial rows.

Therefore a normal Python exception, CUDA failure, or native child crash should
close the child's pipe, allow the family runner to write its footer, and
produce a nonzero family-runner status. The interrupted per-run log has no
`Finish UTC` or `Exit status` footer. This indicates that the family runner did
not regain control to process normal child termination, or was itself
terminated before doing so.

The original wrapper:

- used `set -euo pipefail`, and the tmux command also used `pipefail`;
- captured wrapper stdout and stderr with `2>&1 | tee -a`;
- would not mask a failed left-side pipeline command because `pipefail` was
  enabled;
- had no `ERR`, `EXIT`, `HUP`, `TERM`, or `INT` trap;
- did not explicitly record the wrapper/pipeline exit status;
- had no current phase or experiment context available to a trap;
- did not force unbuffered Python output;
- did not set tmux `remain-on-exit` (the global window option was `off`);
- consequently allowed the pane/session to disappear when its sole command
  exited.

The family runner's footer should have been written for any normal child exit.
The original shell wrapper had no footer of its own.

## Operational behavior audit after hardening

| Desired behavior | Result |
|---|---|
| Exactly one training process at a time | Already present; unchanged sequential runner. |
| `set -Eeuo pipefail` | Added `-E` to wrapper and tmux shell. |
| Line-buffered or unbuffered Python | `PYTHONUNBUFFERED=1` exported. |
| stdout and stderr captured | Already captured by family runner and outer `2>&1 | tee`; unchanged. |
| Exact command logged before execution | Already present in family runner; unchanged. |
| Current experiment recorded | Already in per-run header; wrapper trap context now infers the newest active per-run log name. |
| Start and end timestamps | Already per run; timestamped wrapper and phase events added. |
| Exit code recorded | Already per normal run; ERR and EXIT wrapper statuses added. |
| ERR command and line recorded | Added. |
| Final wrapper status recorded | EXIT trap added. |
| HUP, TERM, INT explicitly logged | Added with conventional statuses 129, 143, and 130. |
| Stop on first failure | Already enforced; unchanged. |
| Completed rows skipped | Already enforced; confirmed. |
| Partial rows block | Already enforced; confirmed. |
| Pane remains inspectable | Per-window `remain-on-exit on` added to the tmux creation command. |
| Scientific commands unchanged | Confirmed by diff inspection. |

The new wrapper-status path is paired with the combined log under `logs/` and
is created only when a future launcher actually runs. No status file was
created during this inspection.

## Root-cause assessment

Confidence describes the available evidence, not certainty.

| Hypothesis | Supporting evidence | Evidence against / limitation | Confidence | Next confirmation test |
|---|---|---|---|---|
| A. GPU/CUDA failure | GPU work was active; inaccessible kernel logs could contain an Xid. | No CUDA/Xid text; driver is currently healthy; three immediately preceding runs completed. | Low | Obtain privileged kernel/NVIDIA logs for 04:10–04:25 and GPU telemetry if retained. |
| B. CUDA OOM | Would terminate/fail a training operation. | No CUDA OOM text; 46 GiB L40; same model shape completed three prior seeds; ordinary CUDA OOM should return control and produce footer/error. | Low | Inspect privileged kernel/application logs and retained GPU-memory telemetry. |
| C. System RAM OOM | Whole process hierarchy could be killed; host has no swap. | No visible OOM record; current RAM is mostly free; historical journal is inaccessible. | Low to moderate | Ask an administrator for kernel OOM/cgroup records for the interval. |
| D. Disk exhaustion | Could prevent log/footer and artifact writes. | Filesystem currently has 102 GiB and abundant inodes; writes succeeded seconds before interruption; no ENOSPC text. | Very low | Check historical filesystem monitoring, if available. |
| E. Python exception | Possible in trainer code. | stderr was captured; no traceback; a normal exception should yield EOF, footer, exit status, and family error. | Low | Future single recovery pilot with unbuffered output and hardened wrapper. |
| F. Native crash/segfault | Could abruptly end the trainer without Python traceback. | Parent should still observe EOF/status and write footer; no segfault text; coredump tooling unavailable. | Low | Administrator checks coredump inventory/audit/kernel logs. |
| G. Wrapper/family exit-status loss | Original wrapper did not print a final wrapper status. | Both shell layers had `pipefail`; family runner explicitly records normal child statuses. A failed left pipeline command was not semantically masked. | Low as root cause; high as visibility defect | Exercise a non-training failure injection in a separate test harness and confirm traps/status propagation. |
| H. SIGHUP/SIGTERM/external termination | Trainer, runner, wrapper, and tmux session all vanished; no family footer; no next run. This best fits termination of the supervising hierarchy/session. | No signal is recorded and visible journals are incomplete. | Moderate | Administrator audit/session/cgroup logs; future signal traps and persistent pane/status log. |
| I. Accidental tmux-session termination | Explains loss of the entire pane hierarchy and absent footer. Login accounting shows pane end at the incident time. | No relevant command in narrow shell history, which is incomplete and untimestamped. | Moderate, within H | Administrator/user audit logs or tmux server logs if retained; future persistent wrapper records. |
| J. Server reboot/administrator action | Administrator action could kill a session without experiment logs. | Reboot is ruled out by boot time, uptime, and `last -x`; no direct evidence of administrator action. | Reboot: very low; targeted action: low/unknown | Administrator system/session/audit logs for the interval. |

### Primary assessment

The leading hypothesis is external termination of the tmux pane or its process
hierarchy, with accidental session termination one plausible mechanism.
Confidence is **moderate**, not high, because no signal or privileged system
event is available. The absence of a traceback must not be treated as proof
against every application, CUDA, or native failure; it is most informative
when combined with the missing family-runner footer.

## Minimal hardening performed

Only `scripts/launch_knn_by_hidden_dim_factorial_tmux.sh` was modified:

- `set -Eeuo pipefail` in wrapper and tmux command;
- `PYTHONUNBUFFERED=1`;
- phase/current-experiment context;
- timestamped wrapper/phase status records;
- ERR command, line, and status logging;
- EXIT status logging;
- explicit HUP/TERM/INT logging;
- a paired wrapper-status file under `logs/`;
- per-window tmux `remain-on-exit on`.

No family-runner, training, model, configuration, or artifact file was
modified. There are no retries or checkpoint-resume behavior.

## Validation

- `bash -n scripts/launch_knn_by_hidden_dim_factorial_tmux.sh`: pass (exit 0).
- `git diff --check`: pass (exit 0).
- Evolve live status: 24 total, 18 complete, 1 partial, 5 missing, 5 runnable.
- Evolve dry run: refused with exit 2 because k6/h32/seed42 is partial and
  missing the three required final artifacts.
- Static live status: 24 total, 12 complete, 0 partial, 12 missing, 12 runnable.
- Static dry run: selected exactly 12 missing rows and printed
  `DRY RUN: no training was executed.`
- The partial blocker remains active.

## Scientifically clean recovery recommendation (not executed)

1. In a later task, preserve the partial run under a clearly documented
   archival path.
2. Before removing it from the canonical path, record original path,
   timestamps, cryptographic hashes, artifact inventory, and interruption
   reason. Do not interpret or resume the checkpoint during preservation.
3. Only after verified preservation, free the canonical output path.
4. Rerun the exact k6/h32/seed42 experiment from epoch 1 as one recovery pilot.
5. Do not resume the epoch-7 checkpoint unless full optimizer and RNG-state
   recovery is proven; the present artifact inventory does not establish that.
6. Verify every required artifact and all prediction-derived metrics.
7. Only after that pilot passes, launch the remaining 17 runs sequentially.

No repository utility for safely archiving a partial experiment was found.
Creating and validating one is recommended as a separate later task. No
recovery launcher was created here.
