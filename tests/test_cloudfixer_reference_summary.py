import copy
import pytest


def fixture():
    config=dict(campaign='a',phase='smoke',manifest=dict(registration=dict(smoke_ids=[1,2],corruptions=['uniform'],native_reference_repeats=3)))
    rows=[]
    for readout in ('terminal','pm'):
        for r,correct in enumerate((1,2,1)):
            rows.append(dict(config=dict(method='cloudfixer',operational_repeat=r),seed=0,readout=readout,status='complete',
                             n_examples=2,n_correct=correct,corruptions=dict(uniform=dict(n_examples=2,n_correct=correct))))
    return dict(campaign='a',rows=rows),config


def test_reference_summary_keeps_each_repeat_and_uses_sample_sd_without_ensembling():
    from scripts.summarize_cloudfixer_reference import reference_summary
    analysis,config=fixture();rows=reference_summary(analysis,config)
    assert len(rows)==2
    for r in rows:
        assert r['accuracy_pct_by_repeat']==[50.,100.,50.]
        assert r['mean_accuracy_pct']==pytest.approx(200/3)
        assert r['sample_sd_pp']==pytest.approx(28.8675134595)
        assert r['seed']==0 and r['n_examples_per_repeat']==2


@pytest.mark.parametrize('case',['partial','coverage','identity','duplicate'])
def test_reference_summary_rejects_incomplete_or_unmatched_repetitions(case):
    from scripts.summarize_cloudfixer_reference import reference_summary
    a,c=fixture()
    if case=='partial':a['rows'][0]['status']='partial'
    elif case=='coverage':a['rows'][0]['corruptions']['uniform']['n_examples']=3
    elif case=='identity':c['campaign']='other'
    else:a['rows'].append(copy.deepcopy(a['rows'][0]))
    with pytest.raises(ValueError):reference_summary(a,c)
