"""Independent CPU recount of the frozen halfmix study; raw bytes stay immutable."""
import argparse,csv,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('old_accuracy',REPO/'knowledge/evidence/decoded_physical_accuracy_20261104/analyze_accuracy.py')
ar=importlib.util.module_from_spec(spec);spec.loader.exec_module(ar)
from research_artifacts import CORRUPTIONS
REMOTE=Path('/content/studies/gsd-lowband-halfmix-20261107')

def load_run(path,root):
 b=ar.open_bundle(path)
 try:
  assert {'command.txt','config.json','environment.txt','stdout.log','summary.csv','per_corruption.csv','notes.md'}<=b.names
  c=b.json('config.json')
  assert c['diagnostic_complete'] and c['status']=='partial' and c['backbone']=='pointmae'
  assert c['seed']==0 and c['severity']==5 and c['batch_size']==2 and c['num_classes']==40
  assert c['guidance_protocol']==dict(gamma=.03,eta=.003,p=.95,total=100,scd_normalize=False)
  assert c['lion_state_before']==c['lion_state_after'] and c['classifier_state_before_sha256']==c['classifier_state_after_sha256']
  assert c['backend_policy']=='controlled' and not c['freeze_decoder_residual']
  common={k:c[k] for k in ('executed_source_files','native_extensions','native_bootstrap','asset_manifest_sha256','dataset_provenance','guidance_protocol','runtime','lion_state_before','classifier_state_before_sha256')}
  cases={};counts={};costs=[];projections=[]
  assert c['asset_manifest_sha256']==ar.identity(root/'source/asset_manifest.json')['sha256']
  selection=root/'operator'/Path(c['index_manifest']).name
  assert b.read('selection_manifest.json')==selection.read_bytes()
  assert c['index_manifest_sha256']==ar.identity(selection)['sha256']
  assert c['selected_original_ids']==json.loads(selection.read_text())['indices']
  manifest=json.loads((root/'source/asset_manifest.json').read_text())
  assert c['asset_manifest']==manifest
  assert hashlib.sha256(b.read('source_snapshot.zip')).hexdigest()==manifest['source_snapshot_sha256']
  for name,entry in c['executed_source_files'].items():
   if name.startswith('assets/'):assert entry==manifest['files'][name[len('assets/'):]]
   else:
    assert name.startswith('repo/');member='source_snapshot/'+name[len('repo/'):]
    data=b.read(member);assert entry==manifest['files'][member]==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
  csvrows=list(csv.DictReader(__import__('io').StringIO(b.read('per_corruption.csv').decode())))
  assert len(csvrows)==len(c['corruptions'])
  for corruption in c['corruptions']:
   parts=[]
   for batch in range(c['count']//2):
    prefix='batches/%s/batch%04d/'%(corruption,batch)
    p=ar.arrays(b,prefix+'predictions.npz');pm,terminal=ar.validate_predictions(p)
    assert p['label'].shape==(2,) and p['label'].dtype.kind in 'iu' and ((p['label']>=0)&(p['label']<40)).all()
    ids=c['selected_original_ids'][batch*2:batch*2+2];assert p['index'].tolist()==ids and p['trajectory_logits'].shape==(2,4,40)
    assert p['candidate_loop_counts'].tolist()==([9,18,27,35] if corruption=='background' else [2,4,6,8])
    d=b.json(prefix+'diagnostics.json');t=d['trajectory'];ref=ar.arrays(b,prefix+'reference.npz')
    assert t['complete'] and t['probe_rng_unchanged'] and t['probe_buffers_unchanged']
    assert ar.tensor_identity(ref['prepared_x'])==d['prepared_x_sha256']
    cache=Path(d['paired_state']['path']).relative_to(REMOTE);cache_id=ar.identity(root/cache)
    assert cache_id=={k:d['paired_state'][k] for k in ('bytes','sha256')}
    with np.load(root/cache,allow_pickle=False) as saved:
     metadata=json.loads(str(saved['metadata'].item()))
     assert metadata['input_sha256']==d['prepared_x_sha256'] and metadata['rng_after_sha256']==t['initial_rng_sha256']
     for key in ('shape_latent','local_latent','noise'):
      assert ar.tensor_identity(saved[key])==metadata['tensor_hashes'][key]==t['initial_hashes'][key]
     assert np.array_equal(saved['local_latent'].reshape(2,2048,4)[:,:,:3],ref['initial_h0_xyz'])
    for entry in list(c['native_extensions'].values())+list(c['native_bootstrap']['bindings'].values()):
     assert ar.identity(root/'native_payload'/str(entry['path']).lstrip('/'))=={k:entry[k] for k in ('bytes','sha256')}
    assert d['scheduler']['final_alpha_cumprod']==1 and d['scheduler']['config']['set_alpha_to_one'] is True
    if c['arm']!='control':
     spectral=t['decoded_spectral']
     assert spectral['loss_gradient_mode']=='full_decoder_gradient' and spectral['decoder_jacobian_in_loss']
     assert spectral['reference_xyz_sha256']==d['prepared_x_sha256']
     assert [r['loop_count'] for r in spectral['records']]==([9,18,27] if corruption=='background' else [2,4,6])
     assert all(r['decoded_geometry_reused'] and not r['classifier_gradients'] for r in spectral['records'])
    first=ar.arrays(b,prefix+'decoded_first_checkpoint.npz')['decoded_xyz']
    identity=dict(scheduler=d['scheduler'],initial=t['initial_hashes'],initial_rng=t['initial_rng_sha256'],
     paired=d['paired_state']['sha256'],prepared=d['prepared_x_sha256'],input_rng=d['input_rng_before_sha256'],output_rng=d['reconstruction_rng_after_sha256'],
     reference={k:ar.tensor_identity(v) for k,v in ref.items()})
    graph=b.json(prefix+'graph.json') if prefix+'graph.json' in b.names else None
    if graph is not None:
     assert graph['reference_xyz_sha256']==d['prepared_x_sha256'] and len(graph['diagnostics'])==2
     basis=ar.arrays(b,prefix+'physical_bases.npz')
     loss=0.
     for i in range(2):
      u=basis['basis_%d'%i];g=graph['diagnostics'][i]
      assert u.shape==(2048,g['actual_rank']) and np.isfinite(u).all() and ar.tensor_identity(u)==graph['basis_sha256'][i]
      assert g['requested_rank']==c['lowband_modes'] and g['spectral_profile']=='hard'
      e=first[i].astype(np.float64)-ref['prepared_x'][i].astype(np.float64)
      loss+=float(np.square(u.astype(np.float64).T@e).sum()/(3*2048))
     live_loss=t['decoded_spectral']['records'][0]['raw_loss']
     projections.append(dict(case=corruption+'/'+str(batch),cpu_basis_loss=loss,raw_native_loss=live_loss,
                             absolute_error=abs(loss-live_loss),passed=abs(loss-live_loss)<=1e-6))
     if prefix+'physical_filters.npz' in b.names:
      filters=ar.arrays(b,prefix+'physical_filters.npz')['filters']
      assert [ar.tensor_identity(f) for f in filters]==graph['filter_sha256']
    key=corruption+'/'+str(batch)
    cases[key]=dict(identity=identity,index=p['index'],label=p['label'],logits=p['trajectory_logits'],first=first,pm=pm,terminal=terminal,
      corruption=corruption,batch=batch,physical_metric_identity=graph,reference=ref)
    costs.append(dict(corruption=corruption,batch=batch,**d.get('timing',{}),decoded_components=[dict({k:r[k] for k in ('loop_count','decoder_and_hash_seconds','spectral_loss_seconds') if k in r},**{k:r['update_norms'][k] for k in ('extra_component_vjp_seconds','extra_component_vjp_calls') if k in r.get('update_norms',{})}) for r in t.get('decoded_spectral',{}).get('records',[])]))
    parts.append(p)
   correct=sum(int((p['prediction']==p['label']).sum()) for p in parts)
   row=next(r for r in csvrows if r['corruption']==corruption)
   assert int(row['n_correct'])==correct and int(row['n_examples'])==c['count'] and float(row['accuracy'])==correct/c['count']
   aggregate=ar.arrays(b,corruption+'_predictions.npz')
   for k in ('index','label','prediction','terminal_prediction','logits','trajectory_logits'):
    assert np.array_equal(aggregate[k],np.concatenate([p[k] for p in parts]))
   counts[corruption]=dict(n=c['count'],pm_correct=correct,terminal_correct=sum(int((p['terminal_prediction']==p['label']).sum()) for p in parts),
      reconstruction_seconds=float(row['runtime_seconds']),peak_gpu_memory_mb=float(row['peak_gpu_memory_mb']))
  summary=list(csv.DictReader(__import__('io').StringIO(b.read('summary.csv').decode())))
  assert len(summary)==1 and int(summary[0]['total_examples'])==c['count']*len(c['corruptions'])
  assert int(summary[0]['total_correct'])==sum(v['pm_correct'] for v in counts.values())
  assert int(summary[0]['n_corruptions'])==len(c['corruptions']) and np.isclose(float(summary[0]['macro_accuracy']),np.mean([v['pm_correct']/v['n'] for v in counts.values()]),rtol=0,atol=1e-12)
  assert np.isfinite([c['wall_seconds'],*[v[k] for v in counts.values() for k in ('reconstruction_seconds','peak_gpu_memory_mb')]]).all()
  return dict(config=c,control_identity=common,cases=cases,counts=counts,costs=costs,projections=projections,archive=ar.identity(path))
 finally:b.close()

def analyze(root,out):
 out.mkdir(exist_ok=False);records={};runs={};failed=[]
 for p in sorted((root/'phase_logs').glob('*.json')):
  r=json.loads(p.read_text());records[r['label']]=r
  if 'bundle' not in r:continue
  path=root/Path(r['bundle']).relative_to(REMOTE)
  if not path.is_file():
   broken=root/'broken_archives'/Path(r['bundle']).relative_to(REMOTE/'raw_runs')
   assert broken.is_file() and ar.identity(broken)==r['archive']
   failed.append(dict(label=r['label'],archive=r['archive'],returncode=r['returncode'],acceptance='unsealed or broken ZIP; directory preserved'));continue
  assert ar.identity(path)==r['archive']
  if r['returncode']!=0:
   b=ar.open_bundle(path)
   try:
    config_status=b.json('config.json')['status'];assert config_status in ('failed','partial')
    if config_status=='partial':assert r.get('timed_out'), 'Unexpected nonzero completed worker'
   finally:b.close()
   failed.append(dict(label=r['label'],archive=r['archive'],returncode=r['returncode'],timed_out=r.get('timed_out',False),config_status=config_status));continue
  runs[r['label']]=load_run(path,root)
 report=dict(status='technical_only',failed_runs=failed,accepted_runs=len(runs),repeats={},pilot=None,fresh=None,
  limitation='Single seed development; original numerical gates unchanged. Native variability is not included in conditional cluster intervals.')
 all_projections=[v for run in runs.values() for v in run['projections']]
 report['all_run_acceptance']=dict(case_count=sum(len(v['cases']) for v in runs.values()),basis_checks=len(all_projections),basis_passed=all(v['passed'] for v in all_projections),max_basis_loss_absolute_error=max(v['absolute_error'] for v in all_projections),
  per_run={name:dict(cases=len(v['cases']),basis_checks=len(v['projections']),basis_passed=all(p['passed'] for p in v['projections'])) for name,v in runs.items()})
 for c in ('gaussian','background'):
  required=['repeat_%s_%s_%d'%(c,a,i) for a in ('control','live_control') for i in range(3)]
  if not all(name in runs for name in required):
   report['repeats'][c]=dict(incomplete=True,available=[name for name in required if name in runs]);continue
  groups={a:[runs['repeat_%s_%s_%d'%(c,a,i)] for i in range(3)] for a in ('control','live_control')}
  gate={}
  for a,v in groups.items():gate[a+'_within']=[ar.repeat_gate(v[i],v[j]) for i in range(3) for j in range(i+1,3)]
  gate['cross']=[ar.repeat_gate(a,b) for a in groups['control'] for b in groups['live_control']]
  report['repeats'][c]=gate
 report['repeat_gate_passed']=all(not sets.get('incomplete') for sets in report['repeats'].values()) and all(g['passed'] for sets in report['repeats'].values() if not sets.get('incomplete') for group in sets.values() for g in group)
 report['phase_statuses']={p.name:json.loads(p.read_text()).get('status') for p in (root/'records').glob('*_phase.json')}
 report['calibration_costs']={name:dict(worker_wall_seconds=r['config']['wall_seconds'],outer_wall_seconds=records[name]['wall_seconds'],extra_component_vjp_seconds=sum(v.get('extra_component_vjp_seconds',0.) for case in r['costs'] for v in case['decoded_components']),extra_component_vjp_calls=sum(v.get('extra_component_vjp_calls',0) for case in r['costs'] for v in case['decoded_components'])) for name,r in runs.items() if name.startswith('cal_')}
 calibration=root/'records/calibration_frozen.json'
 if calibration.is_file():
  frozen=json.loads(calibration.read_text());rows=frozen['rows']
  expected={(m,i,c,t) for m in (100,200) for i in (411,672,777,2247) for c in ('gaussian','background') for t in ([9,18,27] if c=='background' else [2,4,6])}
  assert frozen['status']=='frozen' and len(rows)==48 and {(r['modes'],r['id'],r['corruption'],r['checkpoint']) for r in rows}==expected
  for m in (100,200):
   values=[r for r in rows if r['modes']==m]
   assert all(r['scd_norm']>1e-12 and r['gsd_norm']>1e-12 and np.isclose(r['ratio'],r['scd_norm']/r['gsd_norm'],rtol=1e-12,atol=0) for r in values)
   assert frozen['alpha'][str(m)]==float(np.median([r['ratio'] for r in values]))
  for phase in ('pilot','fresh'):
   path=root/('records/'+phase+'_phase.json')
   if path.is_file():assert json.loads(path.read_text())['freeze_identity']==ar.identity(calibration)
  report['calibration']=dict(alpha=frozen['alpha'],ratios_per_mode=24,identity=ar.identity(calibration))
 aliases=[]
 labels=['control','live0','half0','M100','M200']
 if all('pilot_s%d_%s'%(s,a) in runs for s in (0,1) for a in labels):
  frozen=json.loads((root/'records/calibration_frozen.json').read_text());alpha=frozen['alpha']
  outcomes={};rows=[];costs={}
  expected_ids=json.loads((root/'operator/shard0.json').read_text())['indices']+json.loads((root/'operator/shard1.json').read_text())['indices']
  assert len(expected_ids)==16 and len(set(expected_ids))==16
  for a in labels:
   correctness=[];terminal=[];costs[a]=dict(reconstruction_seconds=0.,graph_seconds=0.,worker_wall_seconds=0.,outer_wall_seconds=0.,decoder_and_hash_seconds=0.,spectral_loss_seconds=0.,peak_allocated_mib=0.)
   for s in (0,1):
    r=runs['pilot_s%d_%s'%(s,a)];c=r['config'];assert c['corruptions']==list(CORRUPTIONS) and c['count']==8 and not c['update_norm_probes']
    selection=root/('operator/shard%d.json'%s)
    assert c['selected_original_ids']==json.loads(selection.read_text())['indices']
    assert c['index_manifest_sha256']==ar.identity(selection)['sha256']
    expected_scd=1. if a in ('control','live0') else .5
    assert c['checkpoint_scd_weight']==expected_scd and c['weight']==(.5*alpha[a[1:]] if a.startswith('M') else 0.)
    assert c['lowband_modes']==(None if a=='control' else (200 if a=='M200' else 100))
    base=runs['pilot_s%d_control'%s];ar.check_pairing(base,r)
    correctness.append(np.stack([np.concatenate([v['pm']==v['label'] for key,v in r['cases'].items() if key.startswith(cor+'/')]) for cor in CORRUPTIONS]))
    terminal.append(np.stack([np.concatenate([v['terminal']==v['label'] for key,v in r['cases'].items() if key.startswith(cor+'/')]) for cor in CORRUPTIONS]))
    costs[a]['reconstruction_seconds']+=sum(v['reconstruction_seconds'] for v in r['counts'].values())
    costs[a]['graph_seconds']+=sum(v['graph_build_seconds'] for v in r['costs'])
    for key in ('decoder_and_hash_seconds','spectral_loss_seconds'):
     costs[a][key]+=sum(v[key] for case in r['costs'] for v in case['decoded_components'])
    costs[a]['worker_wall_seconds']+=c['wall_seconds'];costs[a]['outer_wall_seconds']+=records['pilot_s%d_%s'%(s,a)]['wall_seconds']
    costs[a]['peak_allocated_mib']=max(costs[a]['peak_allocated_mib'],max(v['peak_gpu_memory_mb'] for v in r['counts'].values()))
   outcomes[a]=dict(pm=np.concatenate(correctness,axis=1),terminal=np.concatenate(terminal,axis=1))
   assert outcomes[a]['pm'].shape==(15,16)
   rows.extend(dict(arm=a,corruption=c,n=16,pm_correct=int(outcomes[a]['pm'][i].sum()),terminal_correct=int(outcomes[a]['terminal'][i].sum()),
     pm_accuracy_pct=float(outcomes[a]['pm'][i].mean()*100),delta_operational_pp=float((outcomes[a]['pm'][i].astype(float)-outcomes['control']['pm'][i]).mean()*100)) for i,c in enumerate(CORRUPTIONS))
  f100=[]
  for s in (0,1):
   anchor=runs['pilot_s%d_live0'%s]
   for arm in ('half0','M100'):
    for key,case in runs['pilot_s%d_%s'%(s,arm)]['cases'].items():
     a=anchor['cases'][key]['physical_metric_identity'];b=case['physical_metric_identity']
     f100.append(dict(shard=s,arm=arm,case=key,filter_hashes_equal=a['filter_sha256']==b['filter_sha256']))
  report['M100_filter_consistency']=dict(passed=all(v['filter_hashes_equal'] for v in f100),cases=f100)
  live_gates=[ar.repeat_gate(runs['pilot_s%d_control'%s],runs['pilot_s%d_live0'%s]) for s in (0,1)]
  for s in (0,1):
   for key,c100 in runs['pilot_s%d_M100'%s]['cases'].items():
    c200=runs['pilot_s%d_M200'%s]['cases'][key];g100=c100['physical_metric_identity'];g200=c200['physical_metric_identity']
    aliases.append(dict(shard=s,case=key,ids=c100['index'].tolist(),rank100=[v['actual_rank'] for v in g100['diagnostics']],
       rank200=[v['actual_rank'] for v in g200['diagnostics']],support_same=g100['active_vertex_indices']==g200['active_vertex_indices'],
       filter_hashes_equal=[a==b for a,b in zip(g100['filter_sha256'],g200['filter_sha256'])]))
  contrasts={}
  for a in ('M100','M200'):
   for baseline in ('control','live0','half0'):
    contrasts[a+'_vs_'+baseline]=dict(pm=ar.paired_metrics(outcomes[a]['pm'],outcomes[baseline]['pm']),terminal=ar.paired_metrics(outcomes[a]['terminal'],outcomes[baseline]['terminal']))
  contrasts['live0_vs_control']=dict(pm=ar.paired_metrics(outcomes['live0']['pm'],outcomes['control']['pm']))
  contrasts['half0_vs_live0']=dict(pm=ar.paired_metrics(outcomes['half0']['pm'],outcomes['live0']['pm']))
  report.update(status='pilot_complete_descriptive',pilot=dict(ids=expected_ids,summary={a:dict(pm_correct=int(v['pm'].sum()),terminal_correct=int(v['terminal'].sum()),n=240,pm_accuracy_pct=float(v['pm'].mean()*100)) for a,v in outcomes.items()},
    contrasts=contrasts,costs=costs,control_live_gates=live_gates,control_live_gate_passed=all(g['passed'] for g in live_gates),aliases=aliases,
    projections={'s%d_%s'%(s,a):runs['pilot_s%d_%s'%(s,a)]['projections'] for s in (0,1) for a in labels if a!='control'}))
  with (out/'per_corruption.csv').open('w') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 if any(name.startswith('fresh_') for name in runs):
  report['fresh']={name:dict(costs=r['costs'],worker_wall_seconds=r['config']['wall_seconds'],outer_wall_seconds=records[name]['wall_seconds'],
   scope='Fresh B2 encode/noise each arm; two corruptions, one repeat; instrumentation/IO retained; not B32 throughput or accuracy evidence') for name,r in runs.items() if name.startswith('fresh_')}
 (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=report['status'],accepted_runs=report['accepted_runs'],failed_runs=len(failed),repeat_gate_passed=report['repeat_gate_passed'],pilot_summary=report['pilot']['summary'] if report['pilot'] else None),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);args=p.parse_args();analyze(args.root,args.output)
