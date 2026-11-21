import numpy as np
import pytest

from tests.test_cloudfixer_exploration import fixture_units,payload


def implementation():
    from scripts import analyze_cloudfixer_exploration
    return analyze_cloudfixer_exploration


def test_incomplete_coverage_cannot_aggregate_or_rank():
    m=implementation();wanted=fixture_units()
    with pytest.raises(ValueError,match='coverage'):
        m.analyze([dict(unit=wanted[0],data=payload(wanted[0]))],wanted)
    with pytest.raises(ValueError,match='coverage'):
        m.analyze([dict(unit=u,data=payload(u)) for u in wanted]+[dict(unit=wanted[0],data=payload(wanted[0]))],wanted)


def test_paired_logits_entropy_wrong_confidence_and_pm_use_identical_examples():
    m=implementation();wanted=fixture_units();records=[]
    for u in wanted:
        logits=np.zeros((len(u['ids']),9,40),np.float32)
        logits[:,:,2]=1
        logits[:,0,2]=0;logits[:,0,0]=20  # confident wrong intermediate; terminal is correct.
        records.append(dict(unit=u,data=payload(u,logits=logits)))
    result=m.analyze(records,wanted)
    group=result['configurations'][0]
    assert group['readouts']['old4_pm']['accuracy']==1
    assert group['readouts']['all_candidates_pm']['accuracy']==0
    assert group['candidates'][0]['accuracy']==0
    assert group['candidates'][-1]['accuracy']==1
    assert group['candidates'][0]['mean_entropy_wrong'] < group['candidates'][-1]['mean_entropy_correct']
    assert group['candidates'][0]['intermediate_wrong_terminal_correct']==3
    assert group['oracle']['label_dependent'] and not group['oracle']['deployable']
    assert len(group['candidates'])==9 and group['parity_units']==2


def test_entropy_auc_handles_ties_and_calibration_endpoints():
    m=implementation()
    assert m.correctness_auc(np.array([.8,.8]),np.array([True,False]))==.5
    assert m.correctness_auc(np.array([.9,.1]),np.array([True,False]))==1.
    assert m.correctness_auc(np.array([.9,.1]),np.array([True,True])) is None
    bins=m.calibration(np.array([0.,1.,.5]),np.array([False,True,True]))
    assert sum(b['count'] for b in bins)==3


def test_standard_artifacts_have_exact_counts_and_no_partial_summary(tmp_path):
    import csv
    from scripts import run_cloudfixer_exploration as worker
    m=implementation();wanted=fixture_units()
    result=m.analyze([dict(unit=u,data=payload(u)) for u in wanted],wanted)
    worker.write_reports(tmp_path,result,'a'*64,'entropy')
    assert {'notes.md','summary.csv','per_corruption.csv','analysis.json'} <= {p.name for p in tmp_path.iterdir()}
    with (tmp_path/'summary.csv').open() as f:summary=list(csv.DictReader(f))
    with (tmp_path/'per_corruption.csv').open() as f:corruptions=list(csv.DictReader(f))
    assert len(summary)==2 and len(corruptions)==2
    assert all(r['total_examples']=='3' and r['status']=='complete' for r in summary)
    worker.write_reports(tmp_path,dict(status='partial',configurations=[]),'a'*64,'entropy')
    with (tmp_path/'summary.csv').open() as f:assert not list(csv.DictReader(f))
    with (tmp_path/'per_corruption.csv').open() as f:assert not list(csv.DictReader(f))


@pytest.mark.parametrize('zipped',[False,True])
def test_offline_cli_binds_complete_registered_payloads_to_manifest_and_environment(tmp_path,zipped):
    import json
    import zipfile
    from cloudfixer_store import canonical,sha,unit_key
    from scripts import run_cloudfixer_exploration as worker
    m=implementation()
    manifest=dict(registration=worker.registration(),overlay_files={'scripts/run_cloudfixer_exploration.py':'a'*64})
    identity=dict(manifest=manifest,role='entropy');campaign=sha(canonical(identity))
    wanted=worker.units(campaign,worker.registration(),'entropy')
    root=tmp_path/'attempt';root.mkdir()
    (root/'config.json').write_bytes(canonical(dict(identity,campaign=campaign,execution_status='complete')))
    environment=canonical(dict(model_hashes=['d'*64,'e'*64],gamma_verified=True,lion_loaded=False))
    (root/('environment_'+sha(environment)+'.json')).write_bytes(environment)
    for u in wanted:
        values=worker.unpack(payload(u));values.update(source_sha256=sha(canonical(manifest)),environment_sha256=sha(environment))
        (root/('pending_'+unit_key(u)+'.npz')).write_bytes(worker.pack(values))
    source=root
    if zipped:
        source=tmp_path/'attempt.zip'
        with zipfile.ZipFile(source,'w') as archive:
            for p in root.iterdir():archive.write(p,'attempt/'+p.name)
    output=tmp_path/'analysis.json'
    m.main(['--input',str(source),'--output',str(output)])
    result=json.loads(output.read_text())
    assert result['completed_units']==60 and result['configurations'][0]['readouts']['old4_pm']['count']==1920
    if not zipped:
        first=root/('pending_'+unit_key(wanted[0])+'.npz')
        bad=worker.unpack(first.read_bytes());bad['source_sha256']='f'*64;first.write_bytes(worker.pack(bad))
        output.unlink()
        with pytest.raises(ValueError,match='Source manifest'):m.main(['--input',str(source),'--output',str(output)])
        assert not output.exists()
