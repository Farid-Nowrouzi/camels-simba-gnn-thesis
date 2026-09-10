"""Frozen Set Transformer: separate smoke, train-only, and once-only finalize stages.

Scientific commands require the independently recorded protocol SHA-256.
An interrupted test attempt is deliberately blocked from automatic retry.
"""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from src.evaluation.baseline_common import (
    SPLIT_FILENAMES, compute_metrics, sha256_file, write_json, write_prediction_csv,
)
from src.evaluation.run_modern_deepsets import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, GRAD_CLIP_NORM,
    SCHEDULER_FACTOR, SCHEDULER_PATIENCE, MIN_LR, FEATURE_NAMES,
    seed_everything, make_loader, collect_predictions, load_bound_data,
    load_checkpoint, resolve_device, write_train_log,
)
from src.models.set_transformer import SetTransformerRegressor, EXPECTED_PARAMETERS
from src.models.deepsets import count_parameters
from src.training.split_manifest import load_split_manifest

REPO = Path(__file__).resolve().parents[2]
FAMILY_PATH = Path('configs/experiment_families/u1000_top1500_set_transformer.json')
FREEZE_PATH = Path('reports/experiment_registry/u1000_top1500_set_transformer_protocol_freeze.json')
SMOKE_PATH = Path('outputs/smoke/notebook17_set_transformer')
SEEDS = (42, 123, 2025)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def experiment_name(seed):
    if seed not in SEEDS:
        raise ValueError('Seed must be 42, 123, or 2025.')
    return f'set_transformer_u1000_top1500_raw7_final_train700_seed{seed}_none_d16_h2_m32_isab2_pma1'


def load_family(repo=REPO):
    family = read_json(Path(repo) / FAMILY_PATH)
    freeze = read_json(Path(repo) / FREEZE_PATH)
    if family != freeze['family_configuration']:
        raise ValueError('Family differs from frozen configuration.')
    if family['seeds'] != list(SEEDS) or family['expected_parameter_count'] != EXPECTED_PARAMETERS:
        raise ValueError('Invalid frozen family.')
    return family


def expand_family_jobs(family, repo=REPO):
    jobs = []
    for seed in SEEDS:
        info = family['dataset']['split_manifests'][str(seed)]
        expected_path = f'configs/splits/u1000_top1500_none_k8_sparse/seed{seed}_train700.json'
        if info['path'] != expected_path or sha256_file(Path(repo) / expected_path) != info['sha256']:
            raise ValueError('Exact split binding mismatch.')
        raw = read_json(Path(repo) / expected_path)
        manifest = load_split_manifest(Path(repo) / expected_path,
                                       raw['train_ids'] + raw['val_ids'] + raw['test_ids'],
                                       family['dataset']['sha256'], seed)
        if [len(manifest[k + '_ids']) for k in ('train', 'val', 'test')] != [700, 99, 201]:
            raise ValueError('Expected exact 700/99/201 split.')
        jobs.append({'seed':seed, 'top_n':1500, 'experiment_name':experiment_name(seed),
                     'dataset_path':family['dataset']['path'], 'dataset_sha256':family['dataset']['sha256'],
                     'split_manifest_path':info['path'], 'split_manifest_sha256':info['sha256']})
    return jobs


def verify_freeze(expected_hash, repo=REPO):
    repo = Path(repo)
    if not expected_hash or sha256_file(repo / FREEZE_PATH) != expected_hash:
        raise ValueError('Protocol freeze SHA-256 mismatch or missing approval binding.')
    freeze = read_json(repo / FREEZE_PATH)
    if sha256_file(repo / FAMILY_PATH) != freeze['config_sha256']:
        raise ValueError('Family config SHA-256 mismatch.')
    for relative, digest in freeze['implementation_file_hashes'].items():
        if sha256_file(repo / relative) != digest:
            raise ValueError(f'Implementation source SHA-256 mismatch: {relative}')
    family = load_family(repo)
    expand_family_jobs(family, repo)
    return family


def bound_data(job, repo=REPO):
    absolute = dict(job)
    for key in ('dataset_path', 'split_manifest_path'):
        absolute[key] = str(Path(repo) / job[key])
    return load_bound_data(absolute)


def partition_rows(model, data, manifest, partition, device, *, stage):
    allowed = {'train': ('train', 'val'), 'smoke': ('train', 'val'), 'finalize': ('test',)}
    if partition not in allowed.get(stage, ()):
        raise ValueError(f'{stage} cannot evaluate {partition}.')
    return collect_predictions(model, make_loader(data, manifest[partition + '_ids'], False), device)


def run_config(family, job, freeze_hash, device, smoke=False):
    return {**family, **job, 'protocol_freeze_sha256':freeze_hash,
            'family_config_sha256':sha256_file(REPO / FAMILY_PATH),
            'trainable_parameters':EXPECTED_PARAMETERS, 'device':str(device),
            'scientific_run':not smoke, 'training':family['fixed_configuration'],
            'smoke_protocol':{'seed':42,'train_count':16,'val_count':8,'epochs':2} if smoke else None}


def train_model(data, train_ids, val_ids, seed, device, *, smoke=False):
    generator = seed_everything(seed)
    train_loader = make_loader(data, train_ids, True, generator)
    train_eval_loader = make_loader(data, train_ids, False)
    val_loader = make_loader(data, val_ids, False)
    model = SetTransformerRegressor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, min_lr=MIN_LR,
    )
    criterion = nn.MSELoss()
    best_state, best_mse, best_epoch, stale, log = None, float("inf"), 0, 0, []
    for epoch in range(1, (2 if smoke else MAX_EPOCHS) + 1):
        model.train()
        for _, x, mask, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device), mask.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch}.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        train_rows = collect_predictions(model, train_eval_loader, device)
        val_rows = collect_predictions(model, val_loader, device)
        train_metrics, val_metrics = compute_metrics(train_rows), compute_metrics(val_rows)
        val_mse = float(val_metrics["mse"])
        scheduler.step(val_mse)
        improved = val_mse < best_mse
        if improved:
            best_mse, best_epoch, stale = val_mse, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        log.append({"epoch": epoch, "train_mse": train_metrics["mse"],
                    "validation_mse": val_mse, "validation_rmse": val_metrics["rmse"],
                    "validation_mae": val_metrics["mae"], "learning_rate": optimizer.param_groups[0]["lr"],
                    "improved": improved, "patience_counter": stale})
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return model, log, best_epoch, best_mse


def smoke_checks(model, data, ids, device, checkpoint):
    _, x, mask, _ = next(iter(make_loader(data, ids[:8], False)))
    if x.shape != (8, 1500, 7) or mask.dtype != torch.bool:
        raise AssertionError('Smoke requires full [8,1500,7] and boolean mask.')
    x, mask = x.to(device), mask.to(device)
    model.eval()
    with torch.no_grad():
        reference = model(x, mask)
        assert reference.shape == (8, 1) and torch.isfinite(reference).all()
        permutation = torch.randperm(1500, device=device)
        torch.testing.assert_close(reference, model(x[:, permutation], mask[:, permutation]), rtol=1e-6, atol=1e-6)
        padded = torch.cat((x, torch.full((8, 9, 7), float('nan'), device=device)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros(8, 9, dtype=torch.bool, device=device)), dim=1)
        torch.testing.assert_close(reference, model(padded, padded_mask), rtol=1e-6, atol=1e-6)
        restored = SetTransformerRegressor().to(device)
        restored.load_state_dict(load_checkpoint(checkpoint)['model_state_dict'])
        restored.eval()
        torch.testing.assert_close(reference, restored(x, mask), rtol=0, atol=0)
    seed_everything(42)
    left = SetTransformerRegressor().state_dict()
    seed_everything(42)
    right = SetTransformerRegressor().state_dict()
    assert all(torch.equal(left[key], right[key]) for key in left)
    assert count_parameters(model) == EXPECTED_PARAMETERS
    return {'input_shape':list(x.shape), 'mask_dtype':str(mask.dtype), 'output_shape':[8,1],
            'parameter_count':EXPECTED_PARAMETERS, 'permutation_and_padding_tolerance':{'rtol':1e-6,'atol':1e-6},
            'finite_forward_loss_gradients':True,'backward_optimizer_step':True,
            'checkpoint_roundtrip':True,'deterministic_initialization':True,
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None}


def train_run(seed, freeze_hash, device, *, smoke=False):
    family = verify_freeze(freeze_hash)
    job = next(j for j in expand_family_jobs(family) if j['seed'] == seed)
    if smoke and seed != 42:
        raise ValueError('Smoke seed must be 42.')
    destination = REPO / (SMOKE_PATH if smoke else Path('experiments') / job['experiment_name'])
    # Reserve before data loading/training. Partial directories are never overwritten.
    destination.mkdir(parents=True, exist_ok=False)
    config = run_config(family, job, freeze_hash, device, smoke)
    write_json(config, destination / 'config.json')
    write_json({'state':'training', 'scientific_run':not smoke}, destination / 'run_metadata.json')
    data, manifest = bound_data(job)
    if smoke:
        manifest = {'train_ids':manifest['train_ids'][:16], 'val_ids':manifest['val_ids'][:8]}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model, log, epoch, mse = train_model(data, manifest['train_ids'], manifest['val_ids'], seed, device, smoke=smoke)
    checkpoint = destination / 'checkpoints/best_model.pt'
    checkpoint.parent.mkdir()
    torch.save({'model_state_dict':model.state_dict(), 'config':config,
                'best_epoch':epoch,'best_validation_mse':mse}, checkpoint)
    write_train_log(log, destination / 'train_log.csv')
    metrics = {'best_epoch':epoch, 'best_validation_mse':mse, 'test_status':'pending' if not smoke else 'forbidden'}
    for split in ('train', 'val'):
        rows = partition_rows(model, data, manifest, split, device, stage='smoke' if smoke else 'train')
        write_prediction_csv(rows, destination / 'predictions' / SPLIT_FILENAMES[split])
        metrics['validation' if split == 'val' else split] = compute_metrics(rows)
    write_json(metrics, destination / 'metrics.json')
    if smoke:
        write_json(smoke_checks(model, data, manifest['train_ids'], device, checkpoint), destination / 'smoke_checks.json')
    hashes = {str(p.relative_to(destination)):sha256_file(p) for p in
              [destination / 'config.json', checkpoint, destination / 'train_log.csv',
               destination / 'predictions/train_predictions.csv', destination / 'predictions/val_predictions.csv']}
    write_json({'state':'smoke_complete' if smoke else 'trained', 'scientific_run':not smoke,
                'protocol_freeze_sha256':freeze_hash, 'training_artifact_hashes':hashes,
                'completed_training_utc':datetime.now(timezone.utc).isoformat(),
                'test_status':'forbidden' if smoke else 'pending'}, destination / 'run_metadata.json')


def verify_all_trained(family, freeze_hash, repo=REPO):
    records = []
    for job in expand_family_jobs(family, repo):
        destination = Path(repo) / 'experiments' / job['experiment_name']
        metadata_path = destination / 'run_metadata.json'
        if not metadata_path.is_file():
            raise RuntimeError('Finalize requires completed training for all three seeds.')
        metadata = read_json(metadata_path)
        if (metadata.get('state') not in ('trained', 'finalized') or
                metadata.get('scientific_run') is not True or
                metadata.get('protocol_freeze_sha256') != freeze_hash):
            raise RuntimeError('Finalize requires compatible completed training for all three seeds.')
        hashes = metadata.get('training_artifact_hashes', {})
        required = ('config.json', 'checkpoints/best_model.pt', 'train_log.csv',
                    'predictions/train_predictions.csv', 'predictions/val_predictions.csv')
        for relative in required:
            if not (destination / relative).is_file() or sha256_file(destination / relative) != hashes.get(relative):
                raise ValueError(f'Training artifact hash mismatch: {destination / relative}')
        config = read_json(destination / 'config.json')
        for key, value in {**family, **job, 'protocol_freeze_sha256':freeze_hash,
                           'family_config_sha256':sha256_file(Path(repo) / FAMILY_PATH),
                           'scientific_run':True, 'training':family['fixed_configuration']}.items():
            if config.get(key) != value:
                raise ValueError(f'Incompatible run config: {key}')
        checkpoint = load_checkpoint(destination / 'checkpoints/best_model.pt')
        if checkpoint.get('config') != config:
            raise ValueError('Checkpoint configuration mismatch.')
        model = SetTransformerRegressor()
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        records.append((job, destination, metadata, model))
    return records


def finalize(freeze_hash, device):
    family = verify_freeze(freeze_hash)
    records = verify_all_trained(family, freeze_hash)
    for job, destination, metadata, model in records:
        if metadata['state'] == 'finalized':
            continue
        if (destination / 'test_evaluation_started.json').exists() or (destination / 'predictions/test_predictions.csv').exists():
            raise RuntimeError('Test evaluation already attempted; automatic retry is forbidden.')
        data, manifest = bound_data(job)
        # Exclusive persistent claim precedes any test forward, including concurrent invocations.
        with (destination / 'test_evaluation_started.json').open('x', encoding='utf-8') as handle:
            json.dump({'utc':datetime.now(timezone.utc).isoformat(), 'protocol_freeze_sha256':freeze_hash}, handle)
        rows = partition_rows(model.to(device), data, manifest, 'test', device, stage='finalize')
        write_prediction_csv(rows, destination / 'predictions/test_predictions.csv')
        metrics = read_json(destination / 'metrics.json')
        metrics.update(test=compute_metrics(rows), test_status='evaluated_once')
        write_json(metrics, destination / 'metrics.json')
        metadata.update(state='finalized', test_status='evaluated_once',
                        finalized_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata, destination / 'run_metadata.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('smoke', 'train', 'finalize'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--freeze-sha256', help='Required independently recorded freeze digest for scientific stages.')
    args = parser.parse_args()
    if args.mode != 'smoke' and not args.freeze_sha256:
        parser.error('Scientific stages require --freeze-sha256.')
    if args.mode == 'finalize' and args.seed is not None:
        parser.error('Finalize gates the complete three-seed family; do not select a seed.')
    if args.mode == 'smoke' and args.seed not in (None, 42):
        parser.error('Smoke uses seed 42 only.')
    if args.mode == 'train' and args.seed is None:
        parser.error('Train requires --seed.')
    digest = args.freeze_sha256 or sha256_file(REPO / FREEZE_PATH)
    device = resolve_device(args.device)
    if args.mode == 'finalize':
        finalize(digest, device)
    else:
        train_run(args.seed or 42, digest, device, smoke=args.mode == 'smoke')


if __name__ == '__main__':
    main()
