import io
import json

import numpy as np
import pytest

from cloudfixer_store import canonical, sha, unit_key, load_units, MountStore, Budget
from scripts import run_cloudfixer_transfer as old


def implementation():
    from scripts import run_cloudfixer_exploration
    return run_cloudfixer_exploration


def fixture_units(role='entropy'):
    m = implementation()
    reg = m.registration()
    reg = dict(reg, development_ids=[1, 3, 5], corruptions=['gaussian'], batch_size=2)
    return m.units('a'*64, reg, role)


def payload(unit, *, logits=None):
    m = implementation()
    n = unit['config']['steps']
    loops = list(range(1, n+1)) if unit['role'] == 'entropy' else [n//4, n//2, 3*n//4]
    grid = [round(80*(n-i)/n) for i in range(n+1)]
    count = len(loops)+1
    if logits is None:
        logits = np.zeros((len(unit['ids']), count, 40), dtype=np.float32)
        logits[:, :, 2] = 3
    sampler = dict(indices=grid, candidate_loops=loops+[n],
        candidate_indices=[grid[i-1] for i in loops]+[0], denoiser_calls=n+1,
        valid_reference_counts=[20]*len(unit['ids']), retained_reference_counts=[19]*len(unit['ids']),
        retained_forward_count=19, input_rng_sha256='c'*64)
    values = dict(unit=json.dumps(unit, sort_keys=True), ids=unit['ids'],
        labels=np.full(len(unit['ids']), 2), logits=logits, seconds=1., peak_mb=5.,
        denoiser_calls=n+1, input_shape=[len(unit['ids']),20,3], sampler=json.dumps(sampler),
        model_hashes=['d'*64,'e'*64], repeated_logits=np.empty((0,)),
        source_sha256='f'*64, environment_sha256='b'*64,
        parity=json.dumps(dict(status='passed', old_archive_sha256=m.PRIOR_ARCHIVE_SHA,
            old_unit_key='unit_'+ '9'*64+'.json', max_abs=0., argmax_changes=0,
            pm_argmax_changes=0, pm_max_probability_abs=0.)) if n == 8 else '{}')
    values.update(m.probability_stats(logits))
    stream = io.BytesIO(); np.savez_compressed(stream, **values)
    return stream.getvalue()


def test_registration_preserves_old_ids_and_roles_have_distinct_fixed_coverage():
    m = implementation(); reg=m.registration()
    assert reg['development_ids'] == old.registration()['development_ids']
    assert reg['corruptions'] == old.registration()['corruptions']
    assert reg['budget_seconds'] == 28800
    a=m.units('a'*64,reg,'nstep');b=m.units('a'*64,reg,'entropy')
    assert len(a)==420 and len(b)==60
    assert {u['config']['steps'] for u in a} == set(range(6,13))
    assert {u['config']['steps'] for u in b} == {8}
    assert not ({unit_key(u) for u in a} & {unit_key(u) for u in b})
    assert all(u['seed']==0 and len(u['ids'])==32 for u in a+b)
    assert old.seed_for(a[0]) == old.seed_for(dict(a[0],role='entropy'))


def test_dense_shape_entropy_metadata_and_wrong_labels_fail_closed():
    m=implementation();u=fixture_units()[0];data=payload(u)
    p=m.validate_payload(data,u)
    assert p['logits'].shape==(2,9,40)
    assert p['entropy'].dtype==np.float64
    assert np.allclose(p['normalized_entropy'],p['entropy']/np.log(40))
    assert np.allclose(p['probabilities'].sum(-1),1)
    with pytest.raises(ValueError):m.validate_payload(data,dict(u,ids=[8,9]))
    with pytest.raises(ValueError):m.validate_payload(data,u,labels=np.zeros(8,dtype=int))
    with pytest.raises(ValueError):m.validate_payload(payload(u,logits=np.zeros((2,4,40),np.float32)),u)
    values={**p,'entropy':p['entropy']+1};s=io.BytesIO();np.savez_compressed(s,**values)
    with pytest.raises(ValueError):m.validate_payload(s.getvalue(),u)


def test_same_logit_parity_requires_ids_labels_models_rng_and_exact_argmax():
    m=implementation();u=fixture_units()[0];new=m.validate_payload(payload(u),u)
    baseline=dict(new,logits=new['logits'][:,[1,3,5,8]])
    receipt=m.compare_prior(new,baseline)
    assert receipt['max_abs']==0 and receipt['argmax_changes']==0
    for field,value in [('ids',np.array([7,8])),('labels',np.array([1,2])),
                        ('model_hashes',np.array(['x','y'])),('sampler',json.dumps(dict(input_rng_sha256='x')))]:
        with pytest.raises(ValueError):m.compare_prior(new,dict(baseline,**{field:value}))
    changed=baseline['logits'].copy();changed[:,0,0]+=1
    with pytest.raises(ValueError):m.compare_prior(new,dict(baseline,logits=changed))


def test_interrupted_resume_commits_only_valid_tail(tmp_path):
    m=implementation();wanted=fixture_units();store=MountStore(tmp_path/'data')
    calls=[]
    def infer(u):
        calls.append(u['ids'])
        if u==wanted[1]:raise RuntimeError('interrupted')
        return payload(u)
    with pytest.raises(RuntimeError):m.execute_pending(store,'a'*64,wanted,infer,cap=2000)
    assert len(load_units(store,'a'*64))==1
    calls.clear()
    records=m.execute_pending(store,'a'*64,wanted,lambda u:(calls.append(u['ids']) or payload(u)),cap=2000)
    assert calls==[[5]] and len(records)==2


def test_smoke_chooses_n8_and_reopens_with_multiple_existing_units(tmp_path):
    m=implementation();wanted=fixture_units('nstep');store=MountStore(tmp_path/'data')
    from cloudfixer_store import commit_unit
    commit_unit(store,wanted[-1],payload(wanted[-1]))
    budget=Budget(MountStore(tmp_path/'budget'),cap=2000);calls=[];verified=[];opened=[]
    def reopen():opened.append(True);return MountStore(tmp_path/'data')
    gate=m.smoke_gate(store,reopen,'a'*64,wanted,
        lambda u:(calls.append(u) or payload(u)),lambda:verified.append(True),budget=budget)
    assert [u['config']['steps'] for u in calls]==[8,8]
    assert gate['status']=='passed' and opened and verified
    assert len(load_units(store,'a'*64))==2
    m.smoke_gate(reopen(),reopen,'a'*64,wanted,lambda u:pytest.fail('repeat ran again'),
        lambda:None,budget=budget)


def test_prior_parity_rejects_near_tie_pm_flip_despite_candidate_argmax_and_allclose():
    m=implementation();u=fixture_units()[0];current=m.validate_payload(payload(u),u)
    baseline=np.full((len(u['ids']),4,40),-100,dtype=np.float32)
    baseline[:,:2,0]=10;baseline[:,:2,1]=0
    baseline[:,2:,0]=0;baseline[:,2:,1]=10
    dense=current['logits'].copy();dense[:,[1,3,5,8]]=baseline
    dense[:,1,1]+=1e-5
    previous=dict(current,logits=baseline)
    changed=dict(current,logits=dense)
    subset=dense[:,[1,3,5,8]]
    assert np.allclose(subset,baseline,atol=1e-4,rtol=1e-5)
    assert np.array_equal(subset.argmax(-1),baseline.argmax(-1))
    assert not np.array_equal(m.probability_stats(subset)['probabilities'].mean(1).argmax(-1),
        m.probability_stats(baseline)['probabilities'].mean(1).argmax(-1))
    with pytest.raises(ValueError,match='PM'):
        m.compare_prior(changed,previous)


def test_provenance_requires_manifest_and_immutable_environment_models():
    m=implementation();u=fixture_units()[0];p=m.validate_payload(payload(u),u)
    manifest=dict(overlay_files={'scripts/run_cloudfixer_exploration.py':'a'*64})
    environment=canonical(dict(model_hashes=p['model_hashes'].tolist(),gamma_verified=True,lion_loaded=False))
    p=dict(p,source_sha256=np.array(sha(canonical(manifest))),environment_sha256=np.array(sha(environment)))
    m.validate_provenance(p,manifest,environment)
    with pytest.raises(ValueError):m.validate_provenance(p,dict(manifest,extra=True),environment)
    with pytest.raises(ValueError):m.validate_provenance(p,manifest,environment+b' ')
    other=canonical(dict(model_hashes=['1'*64,'2'*64]))
    with pytest.raises(ValueError):m.validate_provenance(dict(p,environment_sha256=np.array(sha(other))),manifest,other)


def test_prior_archive_recomputes_campaign_key_and_rejects_identity_changes(tmp_path):
    import zipfile
    m=implementation()
    old_manifest=dict(registration=old.registration(),official_files={'adapt.py':'1'*64},
        dataset_files={'label.npy':'2'*64},classifier_sha256='3'*64,diffusion_sha256='4'*64,author_commit='5'*40)
    campaign=sha(canonical(dict(manifest=old_manifest,phase='development',selected=None)))
    manifest=dict(old_manifest,registration=m.registration(),prior_manifest_sha256=sha(canonical(old_manifest)))
    unit=m.units('a'*64,m.registration(),'entropy')[0]
    old_unit={k:v for k,v in unit.items() if k!='role'}
    old_unit.update(campaign=campaign,phase='development')
    dense=m.validate_payload(payload(unit),unit)
    metadata=json.loads(str(dense['sampler']))
    metadata.update(candidate_loops=[2,4,6,8],candidate_indices=[70,50,30,0])
    previous=dict(dense,unit=json.dumps(old_unit),logits=dense['logits'][:,[1,3,5,8]],sampler=json.dumps(metadata))
    archive=tmp_path/'prior.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('old/config.json',canonical(dict(manifest=old_manifest,campaign=campaign,phase='development',execution_status='complete')))
        z.writestr('old/pending_'+unit_key(old_unit)+'.npz',m.pack(previous))
    digest=old.file_hash(archive)
    prior=m.PriorArchive(archive,manifest,expected_sha=digest)
    receipt=prior.check(dense,unit)
    assert receipt['old_unit_key']==unit_key(old_unit) and receipt['pm_argmax_changes']==0
    with pytest.raises(ValueError,match='SHA256'):m.PriorArchive(archive,manifest)
    with pytest.raises(ValueError,match='manifest'):m.PriorArchive(archive,dict(manifest,prior_manifest_sha256='0'*64),expected_sha=digest)
    with pytest.raises(ValueError,match='asset'):m.PriorArchive(archive,dict(manifest,classifier_sha256='0'*64),expected_sha=digest)


def test_source_repeat_gate_is_separate_from_registered_units_and_restores_evidence(tmp_path):
    m=implementation();unit=fixture_units()[0];store=MountStore(tmp_path/'data');budget=Budget(store,cap=2000)
    environment=canonical(dict(model_hashes=['d'*64,'e'*64],gamma_verified=True,lion_loaded=False))
    calls=[];verifies=[]
    def infer(source):
        calls.append(source)
        dense=m.unpack(payload(unit))
        dense.update(unit=json.dumps(source),logits=dense['logits'][:,:1],denoiser_calls=0)
        return m.pack(dense)
    gate=m.source_repeat_gate(store,unit,infer,lambda:verifies.append(True),budget=budget,
        source_sha256='f'*64,environment_bytes=environment)
    assert gate['status']=='passed' and len(calls)==2 and verifies
    assert all(u['config']==dict(method='source') for u in calls)
    assert not store.keys('unit_')
    m.source_repeat_gate(store,unit,lambda u:pytest.fail('source reran'),lambda:None,
        budget=budget,source_sha256='f'*64,environment_bytes=environment)
    gate_bytes=store.get(gate['first_payload'])
    assert sha(gate_bytes) in gate['first_payload']
