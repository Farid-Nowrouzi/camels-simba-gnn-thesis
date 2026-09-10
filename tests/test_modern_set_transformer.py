"""Focused CPU architecture, frozen binding, and stage-boundary tests."""
import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.models.set_transformer import SetTransformerRegressor
from src.models.deepsets import count_parameters
from src.evaluation import run_modern_set_transformer as runner
from src.evaluation.baseline_common import build_prediction_rows, compute_metrics, write_prediction_csv, sha256_file
from scripts.build_experiment_registry import classify_family, find_experiment_dirs


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        runner.seed_everything(42)
        self.model = SetTransformerRegressor().eval()
        self.x = torch.randn(3, 13, 7)
        self.mask = torch.ones(3, 13, dtype=torch.bool)
        self.mask[1, 9:] = False

    def test_exact_count_and_independent_blocks(self):
        self.assertEqual(count_parameters(self.model), 11313)
        self.assertIsNot(self.model.blocks[0].inducing, self.model.blocks[1].inducing)
        self.assertEqual(self.model.blocks[0].inducing.shape, (32, 16))
        self.assertEqual(self.model.pool.seed.shape, (1, 16))

    def test_full_top1500_forward(self):
        with torch.no_grad():
            output = self.model(torch.randn(2, 1500, 7), torch.ones(2, 1500, dtype=torch.bool))
        self.assertEqual(output.shape, (2, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_backward_finite_gradients_optimizer_step(self):
        self.model.train()
        before = self.model.projection[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        loss = torch.nn.MSELoss()(self.model(self.x, self.mask), torch.randn(3, 1))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.projection[0].weight))

    def test_permutation_invariance(self):
        for _ in range(5):
            permutation = torch.randperm(13)
            torch.testing.assert_close(self.model(self.x, self.mask),
                                       self.model(self.x[:, permutation], self.mask[:, permutation]),
                                       rtol=1e-6, atol=1e-6)

    def test_mask_and_padding_invariance(self):
        reference = self.model(self.x, self.mask)
        changed = self.x.clone()
        changed[~self.mask] = float('nan')
        torch.testing.assert_close(reference, self.model(changed, self.mask), rtol=0, atol=0)
        for amount in (1, 7, 31):
            x = torch.cat((changed, torch.full((3, amount, 7), float('nan'))), dim=1)
            mask = torch.cat((self.mask, torch.zeros(3, amount, dtype=torch.bool)), dim=1)
            torch.testing.assert_close(reference, self.model(x, mask), rtol=1e-6, atol=1e-6)
        valid = self.mask[1]
        torch.testing.assert_close(reference[1:2], self.model(self.x[1:2, valid],
                                   torch.ones(1, int(valid.sum()), dtype=torch.bool)), rtol=1e-6, atol=1e-6)

    def test_padded_outputs_zero(self):
        x = self.model.projection(self.x)
        for block in self.model.blocks:
            x = block(x, self.mask)
            self.assertTrue(torch.equal(x[~self.mask], torch.zeros_like(x[~self.mask])))

    def test_all_masked_and_nonboolean_rejected(self):
        self.mask[1] = False
        with self.assertRaisesRegex(ValueError, 'all-masked'):
            self.model(self.x, self.mask)
        with self.assertRaisesRegex(TypeError, 'torch.bool'):
            self.model(self.x, self.mask.float())

    def test_no_graph_or_temporal_api(self):
        self.assertEqual(list(inspect.signature(self.model.forward).parameters), ['x', 'mask'])
        with self.assertRaises(TypeError):
            self.model(self.x, self.mask, edge_index=torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            self.model(self.x.unsqueeze(1), self.mask)

    def test_deterministic_initialization_and_checkpoint(self):
        runner.seed_everything(123)
        left = SetTransformerRegressor()
        runner.seed_everything(123)
        right = SetTransformerRegressor()
        self.assertTrue(all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'checkpoint.pt'
            torch.save({'model_state_dict':self.model.state_dict()}, path)
            right.load_state_dict(runner.load_checkpoint(path)['model_state_dict'])
            right.eval()
            torch.testing.assert_close(self.model(self.x, self.mask), right(self.x, self.mask), rtol=0, atol=0)


class ProtocolTests(unittest.TestCase):
    def test_family_and_exact_split_binding(self):
        family = runner.load_family()
        jobs = runner.expand_family_jobs(family)
        self.assertEqual([j['seed'] for j in jobs], [42,123,2025])
        self.assertEqual(family['feature_names'], runner.FEATURE_NAMES)
        for job in jobs:
            manifest = runner.read_json(runner.REPO / job['split_manifest_path'])
            for name in ('static_gcn_u1000_top1500_sparse_train700_seed{seed}_none_h32_l3_mean_mlp_final',
                         'deepsets_u1000_top1500_raw7_final_train700_seed{seed}_none_h32_phi3_mean_statichead'):
                root = runner.REPO / 'experiments' / name.format(seed=job['seed'])
                with (root / 'predictions/train_predictions.csv').open() as handle:
                    ids = [r['universe_id'] for r in csv.DictReader(handle)]
                self.assertEqual(ids, manifest['train_ids'])
            wrong = json.loads(json.dumps(family))
            wrong['dataset']['split_manifests']['42']['path'] = job['split_manifest_path'] + '.wrong'
            with self.assertRaisesRegex(ValueError, 'split binding'):
                runner.expand_family_jobs(wrong)

    def test_freeze_structure_and_hashes(self):
        digest = sha256_file(runner.REPO / runner.FREEZE_PATH)
        runner.verify_freeze(digest)
        freeze = runner.read_json(runner.REPO / runner.FREEZE_PATH)
        for key in ('scientific_hypothesis','architecture','dataset_path','dataset_sha256','split_hashes',
                    'source_commit','branch','freeze_timestamp_utc','test_use_policy','implementation_file_hashes'):
            self.assertIn(key, freeze)
        with self.assertRaisesRegex(ValueError, 'freeze SHA-256'):
            runner.verify_freeze('0' * 64)

    def test_train_and_smoke_cannot_evaluate_test(self):
        with patch.object(runner, 'make_loader') as loader:
            for stage in ('train', 'smoke'):
                with self.assertRaisesRegex(ValueError, 'cannot evaluate test'):
                    runner.partition_rows(None, {}, {}, 'test', 'cpu', stage=stage)
            loader.assert_not_called()
        self.assertFalse(any('test' in name for name in inspect.signature(runner.train_model).parameters))
        self.assertIn('val_mse < best_mse', inspect.getsource(runner.train_model))

    def test_finalize_missing_three_seed_state_before_data(self):
        with patch.object(runner, 'verify_all_trained', side_effect=RuntimeError('all three seeds')), \
             patch.object(runner, 'bound_data') as data, patch.object(runner, 'partition_rows') as evaluate:
            with self.assertRaisesRegex(RuntimeError, 'all three seeds'):
                runner.finalize(sha256_file(runner.REPO / runner.FREEZE_PATH), torch.device('cpu'))
            data.assert_not_called()
            evaluate.assert_not_called()
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            family = runner.load_family()
            with patch.object(runner, 'expand_family_jobs', return_value=runner.expand_family_jobs(family)):
                with self.assertRaisesRegex(RuntimeError, 'all three seeds'):
                    runner.verify_all_trained(family, 'digest', repo)

    def _completed_runs(self, repo, family, digest):
        config_path = repo / runner.FAMILY_PATH
        config_path.parent.mkdir(parents=True)
        config_path.write_bytes((runner.REPO / runner.FAMILY_PATH).read_bytes())
        jobs = runner.expand_family_jobs(family)
        for job in jobs:
            directory = repo / 'experiments' / job['experiment_name']
            directory.mkdir(parents=True)
            config = runner.run_config(family, job, digest, torch.device('cpu'))
            runner.write_json(config, directory / 'config.json')
            checkpoint = directory / 'checkpoints/best_model.pt'
            checkpoint.parent.mkdir()
            torch.save({'config':config,'model_state_dict':SetTransformerRegressor().state_dict()}, checkpoint)
            runner.write_json({'test_status':'pending'}, directory / 'metrics.json')
            for relative in ('train_log.csv','predictions/train_predictions.csv','predictions/val_predictions.csv'):
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic fixture\n')
            required = ('config.json','checkpoints/best_model.pt','train_log.csv',
                        'predictions/train_predictions.csv','predictions/val_predictions.csv')
            runner.write_json({'state':'trained','scientific_run':True,'protocol_freeze_sha256':digest,
                               'training_artifact_hashes':{r:sha256_file(directory/r) for r in required}},
                              directory / 'run_metadata.json')
        return jobs

    def test_three_seed_completion_and_checkpoint_tampering(self):
        family = runner.load_family()
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            jobs = self._completed_runs(repo, family, 'digest')
            with patch.object(runner, 'expand_family_jobs', return_value=jobs):
                self.assertEqual(len(runner.verify_all_trained(family, 'digest', repo)), 3)
                directory = repo / 'experiments' / jobs[2]['experiment_name']
                metadata = runner.read_json(directory / 'run_metadata.json')
                runner.write_json({**metadata,'state':'training'}, directory / 'run_metadata.json')
                with self.assertRaisesRegex(RuntimeError, 'all three seeds'):
                    runner.verify_all_trained(family, 'digest', repo)
                runner.write_json(metadata, directory / 'run_metadata.json')
                with (directory / 'checkpoints/best_model.pt').open('ab') as handle:
                    handle.write(b'tampered')
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                    runner.verify_all_trained(family, 'digest', repo)

    def test_finalize_once_and_interrupted_attempt_blocks_retry(self):
        family = runner.load_family()
        rows = build_prediction_rows(['synthetic_0','synthetic_1'], [0.2,0.4], [0.25,0.35])
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            jobs = self._completed_runs(repo, family, 'digest')
            with patch.object(runner, 'expand_family_jobs', return_value=jobs):
                records = runner.verify_all_trained(family, 'digest', repo)
            with patch.object(runner, 'verify_freeze', return_value=family), \
                 patch.object(runner, 'verify_all_trained', return_value=records), \
                 patch.object(runner, 'bound_data', return_value=({},{})), \
                 patch.object(runner, 'partition_rows', return_value=rows) as evaluate:
                runner.finalize('digest', torch.device('cpu'))
                self.assertEqual(evaluate.call_count, 3)
                runner.finalize('digest', torch.device('cpu'))
                self.assertEqual(evaluate.call_count, 3)
                # Simulate a crash after claiming a test attempt, before completion.
                records[0][2]['state'] = 'trained'
                with self.assertRaisesRegex(RuntimeError, 'already attempted'):
                    runner.finalize('digest', torch.device('cpu'))
                self.assertEqual(evaluate.call_count, 3)

    def test_source_and_config_tampering_rejected(self):
        digest = sha256_file(runner.REPO / runner.FREEZE_PATH)
        actual = runner.sha256_file
        for suffix in ('src/models/set_transformer.py', str(runner.FAMILY_PATH)):
            def tampered(path):
                return 'tampered' if str(path).endswith(suffix) else actual(path)
            with patch.object(runner, 'sha256_file', side_effect=tampered):
                with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                    runner.verify_freeze(digest)

    def test_partial_destination_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / runner.SMOKE_PATH).mkdir(parents=True)
            family = runner.load_family()
            with patch.object(runner, 'REPO', repo), patch.object(runner, 'verify_freeze', return_value=family), \
                 patch.object(runner, 'bound_data') as data:
                with self.assertRaises(FileExistsError):
                    runner.train_run(42, 'digest', torch.device('cpu'), smoke=True)
                data.assert_not_called()

    def test_prediction_schema_and_canonical_metrics(self):
        rows = build_prediction_rows(['LH_0','LH_1'], [0.2,0.4], [0.25,0.35])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'predictions.csv'
            write_prediction_csv(rows, path)
            with path.open() as handle:
                self.assertEqual(next(csv.reader(handle)), ['universe_id','true_omega_m','pred_omega_m',
                                                           'absolute_error','squared_error'])
        metrics = compute_metrics(rows)
        self.assertAlmostEqual(metrics['mae'], 0.05)
        self.assertAlmostEqual(metrics['mse'], 0.0025)
        self.assertAlmostEqual(metrics['rmse'], 0.05)
        self.assertAlmostEqual(metrics['r2'], 0.75)

    def test_smoke_excluded_and_scientific_family_recognized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / runner.SMOKE_PATH
            smoke.mkdir(parents=True)
            (smoke / 'config.json').write_text('{}')
            self.assertEqual(find_experiment_dirs(root), [])
        self.assertEqual(classify_family(runner.experiment_name(42), 'experiments', {}, {}), 'set_transformer')
        self.assertEqual(classify_family('deepsets_test', 'experiments', {}, {}), 'deepsets')

    def test_control_artifact_hashes(self):
        path = runner.REPO / 'reports/reproducibility/notebook17_control_artifact_manifest.csv'
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 42)
        for row in rows:
            self.assertEqual(sha256_file(runner.REPO / row['path']), row['sha256'])


if __name__ == '__main__':
    unittest.main()
