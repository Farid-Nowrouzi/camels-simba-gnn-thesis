# Sparse U1000 full-source manifest schema

Schema: `camels_source_manifest_v1`. Policy: `full_sha256`. Implementation: `src/data/source_manifest.py`. Hash chunk size: 1,048,576 bytes (1 MiB).

## Entry schema

Every entry contains `relative_path`, `source_role`, `universe_id`, `snapshot_id`, `size_bytes`, and the full-content lowercase hexadecimal `sha256`. Roles are `halo_catalogue`, `target_table`, or justified `other`. Catalogue identity is parsed from `LH_<n>_hlist_<snapshot>.list`; its normalized universe is `LH_<integer>` and snapshot is fixed to five decimals. A target entry additionally records `row_count`, `universe_id_column`, and `target_column`.

Paths are relative to the role-specific root in `source_roots`; absolute roots are operational lookup metadata and are not the portable manifest identity. The dataset metadata separately records `source_root_identity`, the target relative path, and target SHA-256.

## Canonical identity

Entries sort by:

1. role order `halo_catalogue`, `target_table`, `other`;
2. numeric universe ID, with non-universe sources first within their role;
3. numeric snapshot ID;
4. relative path.

The canonical payload contains `schema_version`, `source_manifest_policy`, `hash_algorithm`, `hash_chunk_size_bytes`, `sorting_key`, and the ordered entries. It is UTF-8 JSON with recursively sorted field names and separators `(',', ':')`. `manifest_sha256` is SHA-256 over those exact bytes. Entry/count summaries, verification results, and machine-specific absolute roots are excluded from the portable digest.

## Verification

`verify_full_source_manifest` rejects an invalid schema or role, legacy/stat-only policy, noncanonical order, a bad top-level digest, missing files, size or content-digest changes, invalid relative paths, duplicate relative paths, duplicate catalogue universe/snapshot identities, filename/entry identity disagreement, target structure changes, count disagreement, and anything other than exactly one target table when production target provenance is required.

New sparse builds resolve to `full_sha256` and reject `legacy_stat_only`. Sparse fixture builds using an explicit dummy scalar may contain zero external target sources, with verification called in fixture mode. Dense legacy builds retain their historical stat-only builder behavior. Existing dataset loading is unchanged; `classify_source_provenance` reports old metadata as `legacy_unverified_stat_only` or `legacy_unverified_missing` rather than making it unreadable.

## Dataset metadata projection

New sparse metadata records the full manifest plus:

- `source_manifest_policy`;
- `source_manifest_schema_version`;
- `source_manifest_entry_count`;
- `source_manifest_catalogue_count`;
- `source_manifest_target_source_count`;
- `source_manifest_sha256` and backward-name alias `source_manifest_hash`;
- `source_root_identity`;
- `source_manifest_verification`;
- `target_source_relative_path`;
- `target_source_sha256`.
