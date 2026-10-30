"""Integration guard: depth operating points and frozen guidance arms stay separate."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import eval_chamfer_depth as depth
from scripts import eval_multibackbone_guidance as shared


class SeparateDepthAndVariantContracts(unittest.TestCase):
    def args(self, variant='sparse'):
        return ['--preview','--dataset','modelnet40_c','--dataset-root','/data','--label-path','/data/label.npy',
                '--asset-root','/assets','--result-root','/results','--chamfer-experiment',variant,
                '--backbones','pointmae','--pointmae-config','/config.yaml','--pointmae-checkpoint','/model.pth']

    def test_mixed_programmatic_depth_or_probe_overrides_fail_before_bundle_creation(self):
        for variant in ('sparse','semantic','weighted'):
            base=shared.configurations(self.args(variant))[0]
            for update in (dict(chamfer_depth_study=True),dict(normal_reverse_steps=8),dict(background_reverse_steps=35)):
                mixed=dict(copy.deepcopy(base),**update)
                with self.subTest(variant=variant,update=update),redirect_stdout(StringIO()),self.assertRaisesRegex(ValueError,'fixed|depth'):
                    shared.main(['--preview'],configuration_factory=lambda argv:[mixed])
            mixed=copy.deepcopy(base);mixed['trajectory_selection']['normal_loop_counts']=[2,4,6,8]
            with redirect_stdout(StringIO()),self.assertRaisesRegex(ValueError,'fixed|checkpoint'):
                shared.main(['--preview'],configuration_factory=lambda argv:[mixed])

    def test_worker_rejects_mixed_payload_before_asset_or_cuda_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config=shared.configurations(self.args())[0]
            config['chamfer_depth_study']=True
            (Path(directory)/'config.json').write_text(json.dumps(config))
            with patch.object(shared.source,'read_manifest',side_effect=AssertionError('asset access')) as read:
                with self.assertRaisesRegex(ValueError,'fixed|depth'):
                    shared.worker([directory])
                read.assert_not_called()

    def test_cli_does_not_silently_accept_depth_flags_in_variant_scripts(self):
        with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
            shared.configurations([*self.args(),'--normal-reverse-steps','8'])
        with tempfile.TemporaryDirectory() as directory:
            selection=Path(directory)/'ids.json'
            selection.write_text(json.dumps(dict(dataset='modelnet40_c',n_examples=2468,stage='development',indices=list(range(128)))))
            arguments=['--preview','--dataset-root','/data','--label-path','/data/label.npy','--asset-root','/assets',
                       '--result-root','/results','--index-manifest',str(selection),'--paired-state-root','/cache',
                       '--pointmae-config','/config.yaml','--pointmae-checkpoint','/model.pth']
            self.assertEqual(depth.configurations(arguments)[0]['normal_reverse_steps'],5)
            with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                depth.configurations([*arguments,'--chamfer-experiment','semantic'])


if __name__=='__main__':unittest.main()
