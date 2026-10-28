import copy
from pathlib import Path
import tempfile
import unittest

import collect_confirmation_cache as collector


def fixture(root, drive):
    ids = {name: list(range(140)) for name in collector.DATASETS}
    freeze = dict(status='frozen', phase_plan_sha256=collector.PLAN_SHA, conditions=[dict(id=name) for name in ('spectral-candidate','matched-chamfer','original3dd')])
    expected = collector.expected_cache_paths(ids, root)
    records = [dict(local_path=str(path), drive_path=str(drive / path.relative_to(root)), bytes=123, sha256='a'*64) for path in sorted(expected)]
    jobs = []
    for seed in range(3):
        for condition in freeze['conditions']:
            for dataset in collector.DATASETS:
                original = condition['id']=='original3dd'
                subset = [row for row in records if Path(row['local_path']).parts[-4] == 'paired_sensitivity_confirmation_v2_seed%d'%seed and Path(row['local_path']).parts[-3] == dataset] if not original else []
                traces = {}
                for row in subset:
                    path=Path(row['local_path']); batch=expected[path]['batch'];corruption=expected[path]['corruption']
                    traces[corruption+'/batch%04d'%batch] = dict(paired_state=dict(path=row['local_path'],bytes=row['bytes'],sha256=row['sha256']), initial_hashes=dict(noise='x'))
                jobs.append(dict(status='complete',dataset=dataset,seed=seed,condition_id=condition['id'],trial=dict(method='original3dd' if original else 'gsd'),cache_records=subset,traces=traces))
    state=dict(status='complete',stage='confirmation',session=collector.SESSION,phase_plan_sha256=collector.PLAN_SHA,manifest_sha256=collector.MANIFEST_SHA,source_snapshot_sha256=collector.SOURCE_SHA,jobs=jobs,cache_records=records)
    return state,freeze,ids


class CollectorTests(unittest.TestCase):
    def test_exact_180_three_seed_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root,drive=Path(name).resolve()/'root',Path(name).resolve()/'drive'
            state,freeze,ids=fixture(root,drive)
            records,counts=collector.validated_records(state,freeze,ids,root,drive)
            self.assertEqual(len(records),180)
            self.assertEqual(counts,{0:60,1:60,2:60})
            freeze['conditions'].insert(2,dict(id='tuned-chamfer'))
            additions=[]
            for job in state['jobs']:
                if job['condition_id']=='matched-chamfer':
                    row=copy.deepcopy(job);row['condition_id']='tuned-chamfer';additions.append(row)
            state['jobs'].extend(additions)
            self.assertEqual(len(collector.validated_records(state,freeze,ids,root,drive)[0]),180)

    def test_missing_cache_or_wrong_namespace_or_trace_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root,drive=Path(name).resolve()/'root',Path(name).resolve()/'drive'
            state,freeze,ids=fixture(root,drive)
            unfinished=copy.deepcopy(state);unfinished['status']='running'
            with self.assertRaises(ValueError):collector.validated_records(unfinished,freeze,ids,root,drive)
            missing=copy.deepcopy(state);missing['cache_records'].pop()
            with self.assertRaises(ValueError):collector.validated_records(missing,freeze,ids,root,drive)
            wrong=copy.deepcopy(state);wrong['cache_records'][0]['local_path']=wrong['cache_records'][0]['local_path'].replace('seed0','seed3')
            with self.assertRaises(ValueError):collector.validated_records(wrong,freeze,ids,root,drive)
            wrong=copy.deepcopy(state);wrong['jobs'][0]['traces'][next(iter(wrong['jobs'][0]['traces']))]['paired_state']['sha256']='b'*64
            with self.assertRaises(ValueError):collector.validated_records(wrong,freeze,ids,root,drive)

    def test_parent_symlink_and_escaping_drive_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root,drive=Path(name).resolve()/'root',Path(name).resolve()/'drive'
            state,freeze,ids=fixture(root,drive)
            wrong=copy.deepcopy(state);wrong['cache_records'][0]['drive_path']=str(Path(name)/'outside.npz')
            with self.assertRaises(ValueError):collector.validated_records(wrong,freeze,ids,root,drive)
            root.mkdir();(root/'paired_sensitivity_confirmation_v2_seed0').symlink_to(Path(name)/'foreign',target_is_directory=True)
            with self.assertRaises(ValueError):collector.validated_records(state,freeze,ids,root,drive)

if __name__=='__main__':unittest.main()
