"""Pair-preserving native eigen ordering and explicit graph-free source reuse."""
import copy
import importlib
import unittest
from unittest.mock import patch
import torch

class EigenOrderingTests(unittest.TestCase):
    def test_offdiag_sorts_values_with_their_own_vectors_before_GFT(self):
        graph=importlib.import_module('graph_spectral_v2_offdiag')
        class KNN:
            def __init__(self,**kwargs):pass
            def __call__(self,a,b):
                return torch.zeros(1,4,2),torch.tensor([[[0,1],[1,2],[2,3],[3,0]]])
        eigenvalues=torch.tensor([[0.,1000.0002,1000.,2.]])
        basis=torch.eye(4)[None]
        signal=torch.arange(12,dtype=torch.float32).reshape(1,4,3)
        with patch.object(graph,'KNN',KNN,create=True),patch('torch.linalg.eigh',return_value=(eigenvalues,basis)):
            spectrum,actual_U,actual_e=graph.GraphSpectralDNA(k=2,device='cpu')(signal)
        order=torch.tensor([[0,3,2,1]])
        torch.testing.assert_close(actual_e,eigenvalues.gather(1,order))
        torch.testing.assert_close(actual_U,basis.gather(2,order[:,None,:].expand(-1,4,-1)))
        torch.testing.assert_close(spectrum,actual_U.transpose(1,2)@signal)
        self.assertEqual(actual_e.diff(dim=1).ge(0).all().item(),True)

class SourceReuseTests(unittest.TestCase):
    def records(self):
        shared=dict(object_ids=[1,2],labels=[0,1],seed=0,corruptions=['gaussian','background'],batch_guards=[],spectral_details=[])
        old=dict(asset_manifest_sha256='a'*64,source_snapshot_sha256='b'*64,lion_state_before={'x':'fixed'},
            native_extensions={'x':'fixed'},executed_source_files={'repo/graph_spectral_v2_offdiag.py':{'sha256':'c'*64},'repo/tta.py':{'sha256':'d'*64}})
        new=copy.deepcopy(old);new.update(asset_manifest_sha256='e'*64,source_snapshot_sha256='f'*64)
        new['executed_source_files']['repo/graph_spectral_v2_offdiag.py']={'sha256':'1'*64}
        rows=[dict(shared,scdh_spectralz={'stage':'technical','arm':'off'},calibration_binding=old),
              dict(shared,scdh_spectralz={'stage':'technical','arm':'spectral'},calibration_binding=new)]
        proof=dict(old_asset='a'*64,old_snapshot='b'*64,new_asset='e'*64,new_snapshot='f'*64,
            old_graph={'sha256':'c'*64},new_graph={'sha256':'1'*64},reason='graph-free imported-only module; paired eigen ordering repair')
        return rows,proof

    def test_reuse_requires_explicit_exact_import_only_graph_change(self):
        from scripts.analyze_scdh_spectralz import compare_physical
        rows,proof=self.records()
        with self.assertRaises(ValueError):compare_physical(rows)
        self.assertTrue(compare_physical(rows,technical_reuse=proof))
        rows[0]['scdh_spectralz']['arm']='spectral'
        with self.assertRaises(ValueError):compare_physical(rows,technical_reuse=proof)
        rows,proof=self.records();rows[1]['calibration_binding']['executed_source_files']['repo/tta.py']['sha256']='2'*64
        with self.assertRaises(ValueError):compare_physical(rows,technical_reuse=proof)

    def test_audit_only_reuse_checks_exact_revisions_and_active_code(self):
        from scripts.analyze_scdh_spectralz import compare_physical
        from scripts.run_scdh_spectralz import audit_only_source
        rows,proof=self.records()
        proof['audit_only_revisions']={'tta.py':dict(old={'sha256':'d'*64},new={'sha256':'2'*64})}
        rows[1]['calibration_binding']['executed_source_files']['repo/tta.py']={'sha256':'2'*64}
        self.assertTrue(compare_physical(rows,technical_reuse=proof))
        rows[1]['calibration_binding']['executed_source_files']['repo/tta.py']={'sha256':'3'*64}
        with self.assertRaises(ValueError):compare_physical(rows,technical_reuse=proof)
        before='def forward():\n def audit_clean():\n  return 1\n return 3\n'
        audit_change=before.replace('return 1','return 2')
        self.assertEqual(audit_only_source(before,'tta.py'),audit_only_source(audit_change,'tta.py'))
        self.assertNotEqual(audit_only_source(before,'tta.py'),audit_only_source(before.replace('return 3','return 4'),'tta.py'))
        route='class Route:\n def audit(self):\n  return 1\n def gradients(self):\n  return 3\n'
        self.assertEqual(audit_only_source(route,'scdh_spectralz.py'),audit_only_source(route.replace('return 1','return 2'),'scdh_spectralz.py'))
        self.assertNotEqual(audit_only_source(route,'scdh_spectralz.py'),audit_only_source(route.replace('return 3','return 4'),'scdh_spectralz.py'))
