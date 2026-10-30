"""Asset sealing uses only small source/metadata fixtures; no weights are read."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.prepare_chamfer_variants_assets import build_assets, discover_sources, identity


class AssetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repository = self.root / 'repository'
        self.base = self.root / 'base'
        self.prep = self.root / 'preparation'
        for path in (self.repository, self.base, self.prep):
            path.mkdir()
        self.write(self.repository / 'tta_gsd_v2.py', 'value = 2\n')
        self.write(self.repository / 'semantic_trajectory.py', 'value = 3\n')
        self.write(self.base / 'source_snapshot/tta_gsd_v2.py', 'value = 1\n')
        self.write(self.base / 'model_probe_sources/model.py', 'value = 4\n')
        files = {name: identity(self.base / name) for name in (
            'source_snapshot/tta_gsd_v2.py', 'model_probe_sources/model.py')}
        # Intentionally absent: successful sealing proves neither file is opened.
        files['checkpoints/huge.pth'] = {'bytes': 10 ** 10, 'sha256': 'a' * 64}
        files['datasets/data.npy'] = {'bytes': 10 ** 10, 'sha256': 'b' * 64}
        self.base_manifest = {'files': files, 'datasets': {'modelnet40_c': {'identity': 'keep'}},
                              'guidance': {'native_archive': {'sha256': 'c' * 64},
                                           'lion_checkpoint': {'sha256': 'd' * 64}},
                              'source_overlay': {}}
        self.json(self.base / 'manifest.json', self.base_manifest)
        self.base_sha = identity(self.base / 'manifest.json')['sha256']
        self.json(self.prep / 'study_scope.json', {'backbones': ['dgcnn', 'curvenet', 'pointnext', 'pointmae']})
        selected = {}
        for dataset in ('modelnet40_c', 'scanobjectnn_c'):
            selected[dataset] = {'selected': {}}
            for stage in ('smoke', 'development', 'heldout'):
                name = dataset + '_' + stage + '_ids.json'
                self.json(self.prep / name, {'dataset': dataset, 'stage': stage, 'indices': [1, 2]})
                selected[dataset]['selected'][stage] = {'sha256': identity(self.prep / name)['sha256']}
        self.json(self.prep / 'selection_receipt.json', {'datasets': selected})

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def json(self, path, value):
        self.write(path, json.dumps(value))

    def build(self):
        with patch('scripts.prepare_chamfer_variants_assets.subprocess.check_output', return_value=''):
            return build_assets(self.repository, self.base, self.prep, self.base_sha,
                                source_names=['tta_gsd_v2.py', 'semantic_trajectory.py'],
                                source_commit='fake-commit')

    def test_seal_preserves_bulk_and_native_identities_without_opening_weights(self):
        receipt = self.build()
        assets = self.prep / 'assets_v1'
        manifest = json.loads((assets / 'manifest.json').read_text())
        self.assertFalse(receipt['source_dirty'])
        self.assertFalse(manifest['source_dirty'])
        self.assertEqual(manifest['datasets'], self.base_manifest['datasets'])
        for name in ('checkpoints/huge.pth', 'datasets/data.npy'):
            self.assertEqual(manifest['files'][name], self.base_manifest['files'][name])
            self.assertFalse((assets / name).exists())
        self.assertEqual(manifest['guidance']['native_archive'], self.base_manifest['guidance']['native_archive'])
        self.assertEqual(manifest['guidance']['lion_checkpoint'], self.base_manifest['guidance']['lion_checkpoint'])
        self.assertEqual((self.base / 'source_snapshot/tta_gsd_v2.py').read_text(), 'value = 1\n')
        self.assertEqual((assets / 'source_snapshot/tta_gsd_v2.py').read_text(), 'value = 2\n')
        for dataset in ('modelnet40_c', 'scanobjectnn_c'):
            for stage in ('smoke', 'development', 'heldout'):
                self.assertIn('study/' + dataset + '_' + stage + '_ids.json', manifest['files'])
        with zipfile.ZipFile(assets / 'source_snapshot.zip') as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn('source_snapshot/semantic_trajectory.py', archive.namelist())
            self.assertFalse(any(n.startswith(('checkpoints/', 'datasets/')) for n in archive.namelist()))
        self.assertEqual(receipt['source_snapshot'], identity(assets / 'source_snapshot.zip'))
        self.assertEqual(receipt['asset_layout'], 'overlay_only_reuses_exact_base_bulk')
        with zipfile.ZipFile(self.prep / 'chamfer_guidance_variants_v1_delta.zip') as archive:
            self.assertEqual(set(archive.namelist()), {'assets/manifest.json', 'assets/source_snapshot.zip'})

    def test_refuses_existing_outputs_without_changing_them(self):
        self.build()
        before = identity(self.prep / 'assets_v1/manifest.json')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, identity(self.prep / 'assets_v1/manifest.json'))

    def test_refuses_changed_base_or_ids_before_publishing(self):
        self.write(self.base / 'model_probe_sources/model.py', 'tampered\n')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.prep / 'assets_v1').exists())
        self.write(self.base / 'model_probe_sources/model.py', 'value = 4\n')
        self.write(self.prep / 'modelnet40_c_smoke_ids.json', '{}')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.prep / 'assets_v1').exists())

    def test_refuses_unsafe_source_path(self):
        with self.assertRaises(ValueError):
            build_assets(self.repository, self.base, self.prep, self.base_sha,
                         source_names=['../escape.py'], source_commit='fake')
        self.assertFalse((self.prep / 'assets_v1').exists())

    def test_inventory_adds_runtime_code_and_excludes_generated_or_result_files(self):
        manifest = {'files': {'source_snapshot/scripts/old.py': {}, 'source_snapshot/tta_gsd_v2.py': {}}}
        inventory = 'scripts/new.py\0scripts/build/generated.py\0scripts/.hidden.py\0result/large.json\0tests/test.py\0chamfer_guidance_variants.py\0'
        with patch('scripts.prepare_chamfer_variants_assets.subprocess.check_output', return_value=inventory.encode()):
            actual = discover_sources(self.repository, manifest)
        self.assertEqual(set(actual), {'scripts/old.py', 'tta_gsd_v2.py',
                                      'scripts/new.py', 'chamfer_guidance_variants.py'})

    def test_source_dirty_records_actual_executed_source_status(self):
        with patch('scripts.prepare_chamfer_variants_assets.subprocess.check_output',
                   return_value=' M tta_gsd_v2.py\n') as git:
            receipt = build_assets(self.repository, self.base, self.prep, self.base_sha,
                                   source_names=['tta_gsd_v2.py'], source_commit='fake-commit')
        self.assertTrue(receipt['source_dirty'])
        self.assertEqual(receipt['source_git_status'], ' M tta_gsd_v2.py\n')
        self.assertIn('status', git.call_args.args[0])
        self.assertEqual(git.call_args.args[0][-1], 'tta_gsd_v2.py')


if __name__ == '__main__':
    unittest.main()
