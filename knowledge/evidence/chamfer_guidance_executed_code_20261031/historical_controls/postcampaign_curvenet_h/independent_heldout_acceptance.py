"""Local-only acceptance of new Scan CurveNet C/H evidence; partial mode has no scores."""
import argparse, hashlib, json, math, sys, zipfile
from pathlib import Path
import numpy as np
POST=Path(__file__).resolve().parent;PREP=POST.parent
sys.path.insert(0,str(PREP))
from independent_confirmation_acceptance import array_hash,identity,postprepare_rng_hash,write_new
from scripts.analyze_chamfer_variants import _load,_paired_identity,analyze_campaign
from scripts.analyze_scan_trajectory import Bundle
from research_artifacts import CORRUPTIONS
from scan_curvenet_entropy_report import validate_report
HEADS=('curvenet','dgcnn','pointnext','pointmae');ARMS=('baseline','semantic_h_curvenet')


def accept(manifest_path,output,partial=False):
    manifest=json.loads(Path(manifest_path).read_text());assert manifest['status']=='all_requested_raw_bytes_verified' and manifest['stage']=='heldout'
    rows=manifest['records'];raw=[r for r in rows if '/results/heldout/' in r['path'] and r['path'].endswith('.zip')]
    physical=[r for r in rows if '/paired_states/heldout/' in r['path'] and r['path'].endswith('.npz')]
    assert raw and physical and len({r['path'] for r in raw})==len(raw) and len({r['path'] for r in physical})==len(physical)
    if not partial:assert manifest['phase_complete'] is True and len(raw)==24 and len(physical)==90,'Require24heads/90newcaches'
    plan_path=POST/'phase_plan.json';plan=json.loads(plan_path.read_text());root=Path(plan['research_root'])
    assert plan['source_commit']=='95a49977ef7c3124bde5faa57e3ef36b4393ebcc'
    pinned_path=POST/'source_seal_95a4997/assets_v1/manifest.json';pinned=json.loads(pinned_path.read_text())
    assert identity(pinned_path)=={k:plan['manifest'][k] for k in ('bytes','sha256')}
    ids=json.loads((PREP/'scanobjectnn_c_heldout_ids.json').read_text())['indices'];assert len(ids)==140;chunks=[ids[:70],ids[70:]]
    calraw=(POST/'calibration.json').read_bytes();cal=validate_report(json.loads(calraw));coefficient=cal['coefficients']['semantic_h_curvenet']
    assert hashlib.sha256(calraw).hexdigest()=='8e8dd33e15edc17487bbcc1636064bf3de1b4a1fdad5714d6dd777f62e2e5d07'
    native_proof_path=POST/'independent_calibration_acceptance.json';native_proof=json.loads(native_proof_path.read_text())
    assert native_proof['status']=='independent_native_curvenet_H_calibration_accepted' and native_proof['source_commit']==plan['source_commit'] and native_proof['source_manifest']==identity(pinned_path)
    assert {k:native_proof['report'][k] for k in ('bytes','sha256')}==identity(POST/'calibration.json')
    assert cal['calibration_bundles']['scanobjectnn_c']['selected_original_ids']==json.loads((PREP/'scanobjectnn_c_development_ids.json').read_text())['indices']
    caches={};cacheproof=[]
    for r in physical:
        assert identity(r['local_path'])=={k:r[k] for k in ('bytes','sha256')}
        relative=Path(r['path']).relative_to(root);parts=relative.parts
        assert len(parts)==6 and parts[:2]==('paired_states','heldout') and parts[3]=='scanobjectnn_c'
        seed,corruption=int(parts[2].removeprefix('seed')),parts[4];assert seed in (0,1,2) and corruption in CORRUPTIONS
        matching=[i for i,chunk in enumerate(chunks) if hashlib.sha256(np.asarray(chunk,dtype=np.int64).tobytes()).hexdigest()==relative.stem];assert len(matching)==1
        with np.load(r['local_path'],allow_pickle=False) as a:
            m=json.loads(str(a['metadata'].item()));assert m['schema_version']==1 and m['cuda_devices']==1
            assert a['local_latent'].shape==a['noise'].shape==(70,8192,1,1) and a['shape_latent'].shape[0]==70 and a['shape_latent'].shape[2:]==(1,1)
            for name in ('shape_latent','local_latent','noise'):assert a[name].dtype==np.float32 and np.isfinite(a[name]).all() and array_hash(a[name])==m['tensor_hashes'][name]
            assert a['numpy_rng_keys'].dtype==np.uint32 and a['torch_rng_cpu'].dtype==a['torch_rng_cuda_0'].dtype==np.uint8
            assert postprepare_rng_hash(a,m)==m['rng_after_sha256']
        scope=(seed,corruption,matching[0]);assert scope not in {v['scope'] for v in caches.values()}
        caches[r['path']]=dict(scope=scope,metadata=m,identity={k:r[k] for k in ('bytes','sha256')});cacheproof.append(dict(path=r['path'],sha256=r['sha256'],seed=seed,corruption=corruption,batch=matching[0]))
    if not partial:assert {v['scope'] for v in caches.values()}=={(s,c,b) for s in range(3) for c in CORRUPTIONS for b in range(2)}
    print('NEW_CACHES_VERIFIED',len(caches),flush=True)
    runs=[];lookup={};rawproof=[]
    for r in raw:
        assert Path(r['path']).is_relative_to(root) and identity(r['local_path'])=={k:r[k] for k in ('bytes','sha256')}
        with zipfile.ZipFile(r['local_path']) as z:assert z.testzip() is None
        run=_load(r['local_path']);c=run['config'];arm=run['arm'];seed,head=c['seed'],c['backbone'];key=(seed,head,arm)
        assert key not in lookup and seed in (0,1,2) and head in HEADS and arm in ARMS
        assert c['dataset']=='scanobjectnn_c' and c['batch_size']==70 and c['selected_original_ids']==ids and c['selection_manifest']['stage']=='heldout' and c['corruptions']==list(CORRUPTIONS) and c['max_batches']==0
        assert c['asset_manifest']==pinned and c['asset_manifest_sha256']==plan['manifest']['sha256'] and c['guidance_protocol']['eta']==c['guidance_protocol']['gamma']==.01 and c['guidance_protocol']['p']==.95
        for name,expected in run['sources'].items():
            relative='source_snapshot/'+name[5:] if name.startswith('repo/') else name[7:] if name.startswith('assets/') else None
            assert relative and pinned['files'][relative]==expected,'No original0fd C/source mixing'
        bundle=Bundle(r['local_path'])
        try:
            if arm!='baseline':assert bundle.read('calibration_report.json')==calraw and c['semantic_guidance']['head']=='curvenet' and c['semantic_guidance']['kind']=='marginal_entropy' and c['semantic_guidance']['weight']==c['chamfer_experiment']['semantic_weight']==coefficient
            for corruption in CORRUPTIONS:
                for batch in range(2):
                    d=bundle.json('batches/%s/batch%04d/diagnostics.json'%(corruption,batch));g,t=d['guidance_metrics'],d['trajectory'];n=30 if corruption=='background' else 10
                    assert all(math.isfinite(v) for v in g.values()) and g['graph_calls']==g['spectral_loss_calls']==0 and g['guidance_backward_calls']==n and g['chamfer_calls']==2*n
                    assert d['preparation_mode']==('fresh_encode' if arm=='baseline' else 'cached_replay') and d['geometry_shape']==[70,2048,3]
                    if arm!='baseline':
                        sem=t['semantic_guidance'];assert g['prior_forward_calls']==g['semantic_callback_calls']==n and g['semantic_loss_calls']==2 and sem['head']=='curvenet' and sem['reference']['forward_calls']==1 and not sem['reference_votes']
                        assert [v['loop_count'] for v in sem['records']]==([15,23] if n==30 else [5,8]) and t['operation_counts']==dict(probe_decode_forward=3,probe_classifier_forward=12,semantic_reference_forward=1)
                    cache=caches[d['paired_state']['path']];m=cache['metadata'];assert cache['scope']==(seed,corruption,batch) and cache['identity']=={k:d['paired_state'][k] for k in ('bytes','sha256')}
                    assert m['input_sha256']==t['initial_hashes']['input_points'] and m['rng_after_sha256']==t['initial_rng_sha256'] and all(m['tensor_hashes'][name]==t['initial_hashes'][name] for name in ('shape_latent','local_latent','noise'))
        finally:bundle.close()
        lookup[key]=run;runs.append(run);rawproof.append(dict(seed=seed,head=head,arm=arm,sha256=r['sha256']))
    if not partial:assert set(lookup)=={(s,h,a) for s in range(3) for h in HEADS for a in ARMS}
    for (seed,head,arm),run in lookup.items():
        if arm=='baseline':continue
        baseline=lookup[(seed,head,'baseline')];_paired_identity(baseline,run)
        for corruption in CORRUPTIONS:assert np.array_equal(run['cells'][corruption]['labels'],baseline['cells'][corruption]['labels'])
    proof=dict(status='independent_partial_curvenet_heldout_structurally_accepted' if partial else 'independent_full_curvenet_heldout_accepted',raw_heads=len(raw),raw_batches=30*len(raw),new_caches=len(caches),raw_proofs=rawproof,physical_cache_proofs=cacheproof,manifest=dict(path=str(manifest_path),**identity(manifest_path)),phase_plan=dict(path=str(plan_path),**identity(plan_path)),calibration=dict(path=str(POST/'calibration.json'),**identity(POST/'calibration.json')),coefficient=coefficient,source_commit=plan['source_commit'],source_manifest=identity(pinned_path),no_original_C_or_cache_mixing=True,primary='CurveNet own probability-mean C to H; exploratory posthoc heldout140 reuse',limitation='Native execution is not bitwise deterministic. Cached costs do not prove deployment speed. Geometry finiteness follows the verified producing runtime; raw coordinates are not stored.')
    proof['native_seed0_development_calibration_acceptance']=dict(path=str(native_proof_path),**identity(native_proof_path))
    if not partial:
        report=analyze_campaign([r['local_path'] for r in raw],bootstrap_replicates=2000,bootstrap_seed=0);analysis_path=Path(output).with_name(Path(output).stem+'_analysis.json');write_new(analysis_path,report);proof['secondary_four_head_analysis']=dict(path=str(analysis_path),**identity(analysis_path));proof['per_head_analyses']={}
        for head in HEADS:
            view=analyze_campaign([run['input'] for run in runs if run['config']['backbone']==head],bootstrap_replicates=2000,bootstrap_seed=0);vp=Path(output).with_name(Path(output).stem+'_'+head+'.json');write_new(vp,view)
            proof['per_head_analyses'][head]=dict(path=str(vp),**identity(vp),probability_mean_comparisons=[{k:r[k] for k in ('dataset','arm','scope','baseline_accuracy_pct','arm_accuracy_pct','delta_pp','ci95_delta_pp')} for r in view['comparisons'] if r['policy']=='probability_mean'])
        proof['per_seed_integer_counts']=[]
        for head in HEADS:
            for seed in range(3):
                for arm in ARMS:
                    cells=[r for r in report['per_corruption'] if r['backbone']==head and r['seed']==seed and r['arm']==arm and r['policy']=='probability_mean'];assert len(cells)==15 and sum(r['n'] for r in cells)==2100
                    correct=sum(r['correct'] for r in cells);proof['per_seed_integer_counts'].append(dict(head=head,seed=seed,arm=arm,correct=correct,n=2100,accuracy_pct=100*correct/2100))
        proof['uncertainty']='2000/s0 original-object bootstrap retains all3seeds/15corruptions;140objectclusters and6300decisions per head. CIs condition on included seeds. No coefficient selection or accuracy HPO.'
    write_new(output,proof);print(json.dumps(dict(status=proof['status'],raw_heads=len(raw),new_caches=len(caches),output=str(output))),flush=True)
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=Path);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--partial',action='store_true');args=parser.parse_args();accept(args.manifest,args.output,args.partial)
