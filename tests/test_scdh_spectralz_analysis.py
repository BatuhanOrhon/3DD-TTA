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
    cfg.update(run_id='20261101-fixture',paired_state_root='/remote/paired/seed0',asset_manifest_sha256='a'*64,
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
        from research_artifacts import corruption_row
        rows.append(corruption_row(cfg['run_id'],0,corruption,32,correct,1.,100.,'complete',method=cfg['method'],dataset=cfg['dataset']))
    from research_artifacts import summarize
    for name,data in [('per_corruption.csv',rows),('summary.csv',[summarize(rows,cfg['status'])])]:
        stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data);members[name]=stream.getvalue().encode()
    for name in ('environment.txt','stdout.log','notes.md'):members[name]=b'CPU fixture\n'
    members['command.txt']=b'python /assets/source_snapshot/scripts/eval_scdh_spectralz.py --stage technical --arm off --alpha-h .03 --seed 0\n'
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


def spectral_fixture(root,*,zero=False,false_errors=False,calibrated_zero=False):
    from tests.test_chamfer_depth_transfer_analysis import tensor_hash
    members,binding,cache=fixture(root)
    cfg=json.loads(members['config.json']);cfg['scdh_spectralz'].update(arm='spectral',M_max=800,rho=1.)
    cfg['scdh_technical_gradient_only']=True;cfg['graph']=study.GRAPH.copy()
    for corruption in cfg['corruptions']:
        prefix='batches/'+corruption+'/batch0000/'
        d=json.loads(members[prefix+'diagnostics.json']);n=32
        reference=np.zeros((2,2048,3),np.float32)
        arrays=dict(basis=np.repeat(np.eye(2048,dtype=np.float32)[None],2,axis=0),
            style=np.ones((2,128,1,1),np.float32),style_gradient=np.zeros((2,128,1,1),np.float32) if zero else np.ones((2,128,1,1),np.float32),
            local=np.zeros((2,8192,1,1),np.float32),local_gradient=np.ones((2,8192,1,1),np.float32),
            clean=reference+1,repeat_clean=reference+1,reference=reference,eigenvalues=np.zeros((2,2048),np.float32))
        checks=[]
        for i in (0,1):
            direction=np.ones((128,1,1),np.float32)/np.sqrt(np.float32(128));arrays['direction_%d'%i]=direction
            ad=float((arrays['style_gradient'][i]*direction).sum())
            for scale in (.001,.01):
                eps=scale*float(np.linalg.norm(arrays['style'][i]));fd=ad*1000 if false_errors else ad
                checks.append(dict(example=i,scale=scale,epsilon=eps,autograd=ad,loss_plus=100+eps*fd,
                    loss_minus=100-eps*fd,finite_difference=fd,absolute_error=0.,relative_error=0.,passed=True))
        hashes={k:tensor_hash(v) for k,v in arrays.items()}
        fd=dict(checks=checks,direction_policy='normalized_style_gradient',passed=True,relative_tolerance=.1,absolute_tolerance=1e-5,
            rng_before_sha256='a'*64,rng_after_sha256='a'*64,inputs_before={'local':'b'*64,'style':'c'*64},
            inputs_after={'local':'b'*64,'style':'c'*64},physical_hashes=hashes)
        stream=io.BytesIO();np.savez_compressed(stream,**arrays,metadata=np.asarray(json.dumps(fd,sort_keys=True)))
        raw=stream.getvalue();members['native_fd/'+corruption+'.npz']=raw;fd['sha256']=hashlib.sha256(raw).hexdigest()
        graph=dict(profile='offdiag_static_raw_lambda_truncated_l2',beta=2.,reduction='batch sum / (3*N)',pruning={},
            eigenvalues=np.zeros((32,2048)).tolist(),basis_sha256='a'*64,weights_sha256='b'*64,
            cache_file_sha256='c'*64,eigenvalues_sha256='d'*64,M_max=800,effective_spectral_mass=[800.]*32,
            binding={'reference_sha256':tensor_hash(np.zeros((32,2048,3),np.float32))})
        steps=[]
        for step in range(35 if corruption=='background' else 8):
            r={k:[0.]*32 for k in ('local_scd_gradient_norm','style_scd_gradient_norm','style_spectral_gradient_norm',
                'applied_local_spectral_norm','applied_style_scd_norm','applied_style_spectral_norm','applied_local_scd_norm')}
            r.update(loop_count=step+1,alpha_h_local=.03,kappa_z_spectral_effective=0.)
            steps.append(r)
        metrics=dict(graph=graph,steps=steps,global_diffusion=False,original_z_decode=True,
            final_style_update_consumed_by_decode=False,native_finite_difference=fd)
        d['guidance_metrics']={'scdh_spectralz':metrics};members[prefix+'diagnostics.json']=encoded(d)
    members['config.json']=encoded(cfg)
    binding['expected_config']={k:copy.deepcopy(cfg[k]) for k in analysis.BINDING_KEYS}
    members['command.txt']=b'python /assets/source_snapshot/scripts/eval_scdh_spectralz.py --stage technical --arm spectral --modes 800 --rho 1 --alpha-h .03 --seed 0\n'
    return members,binding,cache


def test_native_fd_rejects_zero_physical_gradients(tmp_path):
    members,binding,cache=spectral_fixture(tmp_path,zero=True)
    with pytest.raises(ValueError,match='gradient|nonzero|Nonzero'):
        analysis.analyze_bundle(seal(tmp_path,members),binding,cache)


def test_native_fd_recomputes_errors_instead_of_trusting_pass_flags(tmp_path):
    members,binding,cache=spectral_fixture(tmp_path,false_errors=True)
    with pytest.raises(ValueError,match='finite.difference|FD|derivative|tolerance'):
        analysis.analyze_bundle(seal(tmp_path,members),binding,cache)


def test_complete_nonzero_physical_fd_witness_is_accepted(tmp_path):
    members,binding,cache=spectral_fixture(tmp_path)
    assert analysis.analyze_bundle(seal(tmp_path,members),binding,cache)['status']=='accepted'


def test_missing_physical_fd_clean_vector_is_rejected(tmp_path):
    members,binding,cache=spectral_fixture(tmp_path)
    for corruption in ('gaussian','background'):
        name='native_fd/'+corruption+'.npz'
        with np.load(io.BytesIO(members[name]),allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files if k!='clean'}
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);members[name]=stream.getvalue()
        prefix='batches/'+corruption+'/batch0000/diagnostics.json';d=json.loads(members[prefix])
        d['guidance_metrics']['scdh_spectralz']['native_finite_difference']['sha256']=hashlib.sha256(members[name]).hexdigest();members[prefix]=encoded(d)
    with pytest.raises(ValueError,match='Missing physical FD vectors'):analysis.analyze_bundle(seal(tmp_path,members),binding,cache)


def test_calibrated_application_requires_an_actual_nonzero_finite_update():
    with pytest.raises(ValueError,match='nonzero'):analysis.verify_calibrated_application([{'applied_style_spectral_norm':[0.,0.]}])
    with pytest.raises(ValueError):analysis.verify_calibrated_application([{'applied_style_spectral_norm':[float('inf')]}])
    analysis.verify_calibrated_application([{'applied_style_spectral_norm':[0.,.001]}])


def test_resealed_summary_units_and_command_identity_are_rejected(tmp_path):
    members,binding,cache=fixture(tmp_path)
    original=members['summary.csv']
    rows=list(csv.DictReader(io.StringIO(original.decode())));rows[0]['macro_accuracy']=str(float(rows[0]['macro_accuracy'])*100)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);members['summary.csv']=stream.getvalue().encode()
    with pytest.raises(ValueError,match='Summary|summary'):analysis.analyze_bundle(seal(tmp_path,members),binding,cache)
    members['summary.csv']=original;members['command.txt']=members['command.txt'].replace(b'--arm off',b'--arm scd')
    with pytest.raises(ValueError,match='command|Command'):analysis.analyze_bundle(seal(tmp_path,members),binding,cache)
