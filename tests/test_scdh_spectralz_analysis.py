"""Actual safe ZIP/physical NumPy1 cache acceptance; no trained inference."""
import copy
import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from scripts import analyze_scdh_spectralz as analysis
from scripts import eval_scdh_spectralz as study
from tests.test_chamfer_depth_transfer_analysis import make_fixture,seal,encoded,identity
from tests import test_scdh_spectralz as config_fixtures

pytestmark=pytest.mark.skipif(np.__version__.split('.')[0]!='1',reason='Raw physical cache verifier is NumPy1')


def fixture(root):
    members,_,cache=make_fixture(root,dataset='modelnet40_c',smoke=True)
    original=json.loads(members['config.json'])
    args=config_fixtures.SpectralStudyConfigTests().args(root,count=32,stage='technical')
    current,=study.configurations(args+['--arm','off'])
    cfg=dict(original,**current)
    cfg.update(paired_state_root='/remote/paired/seed0',asset_manifest_sha256='a'*64,
               native_extensions={'knn_cuda':dict(bytes=1,sha256='a'*64,path='/native/knn.so')})
    (cache/'paired-smoke').rename(cache/'paired')
    source=io.BytesIO();sources={}
    with zipfile.ZipFile(source,'w') as archive:
        for name in analysis.CRITICAL:
            raw=('source '+name).encode();archive.writestr('source_snapshot/'+name,raw)
            sources['assets/source_snapshot/'+name]=identity(raw)
    members['source_snapshot.zip']=source.getvalue()
    cfg.update(source_snapshot_sha256=identity(source.getvalue())['sha256'],executed_source_files=sources,
        executed_source_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    members['selection_manifest.json']=(root/'ids.json').read_bytes()
    rows=[]
    for corruption in cfg['corruptions']:
        prefix='batches/'+corruption+'/batch0000/'
        d=json.loads(members[prefix+'diagnostics.json'])
        d['paired_state']['path']=d['paired_state']['path'].replace('paired-smoke','paired')
        for key in ('executed_source_sha256','source_snapshot_sha256','asset_manifest_sha256'):d[key]=cfg[key]
        members[prefix+'diagnostics.json']=encoded(d)
        with np.load(io.BytesIO(members[prefix+'predictions.npz']),allow_pickle=False) as values:
            payload={k:values[k].copy() for k in values.files}
        payload.update(terminal_prediction=payload['prediction'],prediction=payload['probability_mean'])
        stream=io.BytesIO();np.savez(stream,**payload);members[prefix+'predictions.npz']=stream.getvalue()
        correct=int((payload['prediction']==payload['label']).sum())
        rows.append(dict(corruption=corruption,n_examples=32,n_correct=correct,accuracy=correct/32,status='complete'))
    for name,data in [('per_corruption.csv',rows),('summary.csv',[dict(status=cfg['status'],total_examples=64,total_correct=sum(r['n_correct'] for r in rows))])]:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data);members[name]=stream.getvalue().encode()
    for name in ('command.txt','environment.txt','stdout.log','notes.md'):members[name]=b'CPU fixture\n'
    members['config.json']=encoded(cfg)
    binding=dict(slot='technical-off',expected_config={k:copy.deepcopy(cfg[k]) for k in analysis.BINDING_KEYS},required_source_files=sources)
    return members,binding,cache


def test_real_probability_primary_physical_acceptance_and_tamper(tmp_path):
    members,binding,cache=fixture(tmp_path)
    report=analysis.analyze_bundle(seal(tmp_path,members),binding,cache)
    assert report['status']=='accepted' and report['correct']==62
    assert report['outcomes']['terminal'][0][:2]==[False,True]
    assert report['outcomes']['probability_mean'][0][:2]==[True,False]
    members.pop('notes.md')
    with pytest.raises(ValueError,match='standard artifact'):analysis.analyze_bundle(seal(tmp_path,members),binding,cache)


def test_resealed_incomplete_and_policy_tamper_rejected(tmp_path):
    members,binding,cache=fixture(tmp_path)
    prefix='batches/gaussian/batch0000/'
    with np.load(io.BytesIO(members[prefix+'predictions.npz']),allow_pickle=False) as values:payload={k:values[k].copy() for k in values.files}
    payload['prediction']=payload['terminal_prediction']
    stream=io.BytesIO();np.savez(stream,**payload);members[prefix+'predictions.npz']=stream.getvalue()
    with pytest.raises(ValueError,match='primary policy'):analysis.analyze_bundle(seal(tmp_path,members),binding,cache)


def test_zip_symlink_rejected_before_manifest(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as archive:
        info=zipfile.ZipInfo('link');info.external_attr=0o120777<<16;archive.writestr(info,'/tmp')
    with pytest.raises(ValueError,match='Unsafe archive'):analysis.safe_archive(path)
