# U1000 Top1000 source-manifest policy review

Audit date: 2026-08-05. This review inspected the implementation and performed a bounded benchmark over all 15 catalogues from `LH_418`, `LH_131`, and `LH_847` (62,210,047 bytes). It did not hash the full corpus or create a production manifest.

## Current implementation

`build_temporal_dataset` traverses the successfully built in-memory samples and constructs one line per catalogue as:

```text
<snapshot path>\t<stat size>\t<stat mtime_ns>\n
```

It SHA-256 hashes the concatenated lines. Therefore the current hash covers ordered path (and thereby filename), file size, and nanosecond modification time. It does **not** cover inode, header, bounded chunks, complete catalogue content, target-source stat/content, or other filesystem metadata. `targets_csv` is only recorded as a path string elsewhere in metadata. A content change that preserves size and mtime is undetected, and a target CSV can change without changing `source_manifest_hash`.

## Policy comparison

| Policy | Reproducibility | Undetected-change risk | Estimated one-time runtime | Estimated manifest size | Complexity | Production status |
|---|---|---|---|---|---|---|
| A. Current stat-only | Weak-to-moderate under an enforced immutable-filesystem policy | Material: preserved-size/mtime edits and every target-content edit evade detection | Metadata stat traversal; likely seconds | About 0.6–1.0 MB for 5,001 records | Already implemented | Blocker as currently implemented because no immutable-source enforcement is encoded and target content is absent |
| B. Full SHA-256 for 5,000 catalogues plus target source | Strong: byte-exact source identity | Cryptographic residual only | Sample throughput 1,441.6 MiB/s on warm cache; linear 22.149 GB estimate 14.7 s, with a conservative cold-I/O allowance of 1–4 min | About 0.9–1.3 MB | Low: streaming hash per file and one target record | Recommended; not a storage/runtime blocker |
| C. Hybrid stat/header/first-last 1 MiB/selected metadata | Moderate-to-strong for accidental edits, not byte-exact | Interior-only changes can evade detection | Sample read 26.70 MiB in 0.0187 s; likely seconds to tens of seconds for 5,001 files | About 1–3 MB depending on selected metadata | Moderate; chunk and header policy must be versioned | Acceptable only with a documented limitation, but offers little benefit because full hashing is already cheap |

## Recommendation and classification

Use policy B. Stream SHA-256 over each of the 5,000 raw catalogue files and the exact target CSV, persist per-file hashes in deterministic universe/snapshot order, and hash that manifest into dataset metadata. Bind split manifests to the published dataset checksum after the build.

**Current policy classification: production blocker.** The bounded full-content measurement shows that strong provenance is inexpensive relative to the measured 36–70 minute build, while the existing builder has no CLI/config option to select or ingest a full-content source manifest and omits the target source entirely. A separate manually generated manifest would be useful evidence but would not repair the dataset metadata binding unless the builder records and verifies it.
