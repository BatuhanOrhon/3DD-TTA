"""Local frozen140-ID sensitivity confirmation; no inference or policy reselection."""
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import pickle
import re
import sys

import numpy as np

import analyze_sensitivity_local as shared

DATASETS = shared.DATASETS
HEADS = shared.HEADS
POLICIES = shared.POLICIES
SEEDS = (0, 1, 2)
ORIGINAL_PLAN_SHA = '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
SPENT_FLOOR = 155.86660684299568
require = shared.require


def load_context(prep, plan_path, plan_sha):
    """Keep operational recovery separate from the unchanged sealed scientific scope."""
    require(re.fullmatch('[0-9a-f]{64}', plan_sha) is not None, 'Require explicit reviewed plan SHA256')
    require(shared.identity(plan_path)['sha256'] == plan_sha, 'Provided plan SHA mismatch')
    original = shared.read_json(prep / 'campaign_plan.json')
    require(shared.identity(prep / 'campaign_plan.json')['sha256'] == ORIGINAL_PLAN_SHA, 'Original registration changed')
    plan = shared.read_json(plan_path)
    scientific = copy.deepcopy(plan); recovery = scientific.pop('recovery', None)
    for key in ('session', 'endpoint'):
        scientific[key] = original[key]
    scientific['phase_sources']['sensitivity_campaign_controller.py'] = original['phase_sources']['sensitivity_campaign_controller.py']
    require(scientific == original and recovery is not None and recovery['original_plan_sha256'] == ORIGINAL_PLAN_SHA,
            'Recovery changed scientific scope/IDs/assets/objectives/budget')
    require(plan['session'] != original['session'] and plan['endpoint'] != original['endpoint'], 'Expected explicit fresh-runtime recovery')
    context = list(shared.load_context(prep, plan_path, plan_sha))
    controller = context[1]
    require(controller.SESSION == plan['session'] and controller.CAP == 19800, 'Migrated controller ownership/cap changed')
    for dataset in DATASETS:
        ids = context[6][('confirmation', dataset)]['indices']
        require(len(ids) == 140 and not set(ids) & set(context[6][('development', dataset)]['indices']), 'Confirmation/development overlap')
        previous = shared.read_json(prep.parent / 'guidance_component_study_20261025/assets_v1/study' / (dataset + '_heldout_ids.json'))
        require(not set(ids) & set(previous['indices']), 'Previously exposed70 held-out IDs reused')
    return tuple(context)


def frozen_conditions(freeze, accepted, scores, registry, controller, calibration, accepted_identity, plan_sha):
    require(accepted.get('status') == 'accepted' and accepted.get('physical_cache_complete') is True
            and accepted.get('phase_plan_sha256') == plan_sha and accepted.get('selection_policy') == 'probability_mean',
            'Independent physical development acceptance missing')
    require([r['trial_id'] for r in scores] == list(registry) and all(math.isfinite(r['accuracy_pct']) for r in scores),
            'Registered development ranking/order missing')
    expected = shared.freeze_selection(scores, registry, accepted_identity, controller, calibration)
    expected['phase_plan_sha256'] = plan_sha
    require(freeze == expected and freeze['selected_spectral_trial_id'] == accepted['selected_spectral_trial_id']
            and freeze['selected_chamfer_trial_id'] == accepted['selected_chamfer_trial_id'], 'Frozen condition list/ranking differs from registered construction')
    candidate = registry[freeze['selected_spectral_trial_id']]
    require(candidate['family'] in ('combined', 'spectral'), 'Selected candidate is not spectral-containing')
    matched = copy.deepcopy(registry['c-base'])
    matched.update(id='matched-chamfer', reuse=False, eta=candidate['eta'], gamma=candidate['gamma'], p=candidate['p'])
    trials = dict(registry, **{'matched-chamfer': matched})
    result = [dict(id=r['id'], role=r['role'], trial=trials[r['trial_id']]) for r in freeze['conditions']]
    require(3 <= len(result) <= 4 and len({r['id'] for r in result}) == len(result), 'Frozen conditions duplicated')
    return result


def label_references(context, modelnet_labels):
    manifest,assets,selections=context[4],context[3],context[6]
    aligned,proof={},[]
    for dataset in DATASETS:
        path=Path(modelnet_labels) if dataset=='modelnet40_c' else assets/'datasets'/dataset/'label.npy'
        expected=manifest['datasets'][dataset]['files']['label.npy']
        shared.verify(path,expected)
        values=np.load(path,allow_pickle=False)
        count,classes=(2468,40) if dataset=='modelnet40_c' else (581,15)
        require(list(values.shape)==expected['shape'] and str(values.dtype)==expected['dtype']
                and values.dtype.kind in 'iu' and values.size==count and ((values>=0)&(values<classes)).all(),
                'Full sealed label shape/dtype/classes mismatch')
        ids=np.asarray(selections[('confirmation',dataset)]['indices'],dtype=np.int64)
        aligned[dataset]=dict(index=ids,label=values.reshape(-1)[ids])
        proof.append(dict(dataset=dataset,path=str(path),**shared.identity(path)))
    return aligned,proof


def raw_outcomes(path, original, corruptions, policies):
    result = {p: [] for p in (('terminal',) if original else POLICIES)}
    bundle = shared.checked_zip(path)
    try:
        for corruption in corruptions:
            names = sorted(n for n in bundle.namelist() if n.startswith('batches/' + corruption + '/') and n.endswith('/predictions.npz'))
            selected = {p: [] for p in result}
            for name in names:
                with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as data:
                    predictions = {'terminal': data['prediction']} if original else policies(data['trajectory_logits'], terminal=3)
                    for policy in result:
                        selected[policy].append(predictions[policy] == data['label'])
            for policy in result:
                result[policy].append(np.concatenate(selected[policy]))
    finally:
        bundle.close()
    return {p: np.asarray(rows, dtype=bool) for p, rows in result.items()}


def balanced_table(outcomes, conditions, corruptions):
    rows = []
    for scope in ('all15', 'macro14'):
        mask = np.asarray([scope == 'all15' or c != 'background' for c in corruptions])
        for condition in conditions:
            policies = ('terminal',) if condition == 'original3dd' else POLICIES
            for policy in policies:
                values = []
                for seed in SEEDS:
                    cases = [outcomes[(d,h,condition,policy,seed)][mask].mean() for d in DATASETS for h in HEADS]
                    value = 100. * float(np.mean(cases)); values.append(value)
                    rows.append(dict(scope=scope, condition_id=condition, policy=policy, seed=seed, aggregation='seed',
                                     accuracy_pct=value, n_cases=8, per_seed_min_accuracy_pct=value, per_seed_max_accuracy_pct=value))
                rows.append(dict(scope=scope, condition_id=condition, policy=policy, seed=None, aggregation='three_seed_mean',
                                 accuracy_pct=float(np.mean(values)), n_cases=8, per_seed_min_accuracy_pct=min(values), per_seed_max_accuracy_pct=max(values)))
    return rows


def paired_tables(outcomes, corruptions, comparators, bootstrap_replicates=2000):
    """One140-ID draw matrix per dataset is reused jointly across every seed/contrast."""
    require(len(corruptions) == len(set(corruptions)) == 15 and 'background' in corruptions, 'Require all15 corruption ordering')
    require(type(bootstrap_replicates) is int and bootstrap_replicates > 0, 'Invalid bootstrap count')
    rng = np.random.default_rng(20261025)
    draws = {d: rng.integers(0,140,size=(bootstrap_replicates,140)) for d in DATASETS}
    cases, balanced = [], []
    for scope in ('all15', 'macro14'):
        mask = np.asarray([scope == 'all15' or c != 'background' for c in corruptions])
        for comparator in comparators:
            policy = 'terminal' if comparator == 'original3dd' else 'probability_mean'
            per_seed, selected = {}, {}
            for seed in SEEDS:
                effects, group = {}, []
                for dataset in DATASETS:
                    per_object = []
                    for head in HEADS:
                        left = outcomes[(dataset,head,'spectral-candidate','probability_mean',seed)]
                        right = outcomes[(dataset,head,comparator,policy,seed)]
                        require(left.shape == right.shape == (15,140) and left.dtype == right.dtype == bool,
                                'Require paired all15x140 outcomes')
                        left,right = left[mask],right[mask]
                        difference = left.astype(float)-right.astype(float)
                        per_object.append(difference.mean(axis=0)); delta=100.*float(difference.mean())
                        group.append(dict(scope=scope, comparator=comparator, comparator_policy=policy,
                            candidate='spectral-candidate', candidate_policy='probability_mean', seed=seed,
                            dataset=dataset, backbone=head, n_prediction_events=int(mask.sum())*140,
                            delta_pp=delta, corrected_events=int((left & ~right).sum()), harmed_events=int((~left & right).sum()),
                            sign='positive' if delta>0 else 'negative' if delta<0 else 'tied',
                            comparison='whole_policy_different_host' if comparator=='original3dd' else 'same_host_same_readout'))
                    effects[dataset]=np.mean(per_object,axis=0)
                per_seed[seed]=100.*np.mean([effects[d][draws[d]].mean(axis=1) for d in DATASETS],axis=0)
                cases.extend(group);selected[seed]=group
            seed_deltas=[np.mean([r['delta_pp'] for r in selected[s]]) for s in SEEDS]
            for seed in (*SEEDS,None):
                seeds=SEEDS if seed is None else (seed,)
                chosen=[r for s in seeds for r in selected[s]]
                sampled=np.mean([per_seed[s] for s in seeds],axis=0)
                low,high=np.percentile(sampled,[2.5,97.5])
                effects=[np.mean([r['delta_pp'] for r in chosen if r['dataset']==d and r['backbone']==h]) for d in DATASETS for h in HEADS]
                balanced.append(dict(scope=scope,comparator=comparator,comparator_policy=policy,candidate='spectral-candidate',
                    candidate_policy='probability_mean',seed=seed,aggregation='three_seed_mean' if seed is None else 'seed',
                    n_cases=8,n_seeds=len(seeds),n_original_object_clusters=280,n_prediction_events=sum(r['n_prediction_events'] for r in chosen),
                    delta_pp=float(np.mean(effects)),corrected_events=sum(r['corrected_events'] for r in chosen),
                    harmed_events=sum(r['harmed_events'] for r in chosen),positive_cases=int(sum(v>0 for v in effects)),
                    negative_cases=int(sum(v<0 for v in effects)),tied_cases=int(sum(v==0 for v in effects)),
                    per_seed_min_pp=float(min(seed_deltas) if seed is None else seed_deltas[seed]),
                    per_seed_max_pp=float(max(seed_deltas) if seed is None else seed_deltas[seed]),
                    conditional_object_bootstrap_low_pp=float(low),conditional_object_bootstrap_high_pp=float(high),
                    comparison=chosen[0]['comparison']))
    return cases,balanced


def cost_table(rows):
    grouped={}
    for row in rows:
        key=row['dataset'],row['condition_id'],row['seed'],row['batch']
        entry=grouped.setdefault(key,dict(dataset=key[0],condition_id=key[1],seed=key[2],batch=key[3],
            shared_reconstruction_seconds=row['shared_reconstruction_seconds'],classifier_forward_seconds=0.,peak_gpu_memory_mb=0.,
            preparation_mode=row['preparation_mode'],heads=[]))
        require(entry['shared_reconstruction_seconds']==row['shared_reconstruction_seconds']
                and entry['preparation_mode']==row['preparation_mode'], 'Cost heads do not share reconstruction')
        require(row['backbone'] not in entry['heads'], 'Duplicate cost head')
        entry['heads'].append(row['backbone'])
        entry['classifier_forward_seconds']+=row['classifier_forward_seconds']
        entry['peak_gpu_memory_mb']=max(entry['peak_gpu_memory_mb'],row['whole_process_peak_gpu_memory_mb'])
    require(all(set(r['heads'])==set(HEADS) for r in grouped.values()), 'Require four distinct classifier costs per shared reconstruction')
    return [dict(r,heads=len(r['heads'])) for r in grouped.values()]


def cache_rng_hash(cache, metadata):
    require(np.__version__.split('.')[0]=='1', 'Cache RNG verification requires NumPy1; use the pinned CPU analysis environment')
    py,nr=metadata['python_rng'],metadata['numpy_rng']
    state=((py[0],tuple(py[1]),py[2]),(nr[0],cache['numpy_rng_keys'].copy(),*nr[1:]),cache['torch_rng_cpu'].copy(),[cache['torch_rng_cuda_0'].copy()])
    return hashlib.sha256(pickle.dumps(state,protocol=4)).hexdigest()


def validate_physical_caches(records, paired, roots):
    physical=shared.physical_caches(records,roots)
    by_path={row['remote_path']:row for row in physical}
    require(len(by_path)==180 and len(paired)==180, 'Require180 confirmation caches:3seeds x2datasets x15corruptions x2batches')
    for (seed,dataset,corruption,batch),trace in paired.items():
        record=by_path.get(trace['paired_state']['path'])
        require(record is not None and all(record[k]==trace['paired_state'][k] for k in ('bytes','sha256')), 'Raw trace lacks physical-cache receipt')
        if not record['local_verified']:
            continue
        with np.load(record['local_paths'][0],allow_pickle=False) as cache:
            metadata=json.loads(str(cache['metadata'].item())); initial=trace['initial_hashes']
            require(metadata['schema_version']==1 and metadata['cuda_devices']==1 and metadata['input_sha256']==initial['input_points'], 'Cache/input metadata mismatch')
            expected={'metadata','numpy_rng_keys','torch_rng_cpu','torch_rng_cuda_0','shape_latent','local_latent','noise'}
            require(set(cache.files)==expected and set(metadata['tensor_hashes'])=={'shape_latent','local_latent','noise'}, 'Cache schema mismatch')
            for name in ('shape_latent','local_latent','noise'):
                array=cache[name]
                digest=hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()
                require(array.dtype==np.float32 and array.ndim==4 and array.shape[0]==70 and array.shape[1]>0 and array.shape[2:]==(1,1) and np.isfinite(array).all()
                        and digest==metadata['tensor_hashes'][name]==initial[name], 'Cache latent/noise identity mismatch')
            require(cache['local_latent'].shape==cache['noise'].shape==(70,8192,1,1), 'Invalid local latent/noise shape')
            require(cache_rng_hash(cache,metadata)==metadata['rng_after_sha256'], 'Cache post-preparation RNG differs')
            require(all(re.fullmatch('[0-9a-f]{64}',metadata[k]) for k in ('rng_before_sha256','rng_after_sha256')), 'Invalid cache RNG metadata')
    return physical


def validate_classifier_contracts(heads, ledger):
    for (dataset,head),contract in heads.items():
        expected=dict(state_sha256=contract['state'],checkpoint_sha256=contract['checkpoint'],
                      preprocessing=contract['preprocessing'],resolved_model_config=contract['resolved'])
        require(ledger['classifier_contracts'][dataset+'/'+head]==expected, 'Classifier contract differs from cumulative campaign ledger')


def run(args):
    prep=args.prep.resolve();plan_path=args.plan.resolve();output=args.output.resolve()
    require(not output.exists(), 'Require fresh analysis output; preserve existing evidence')
    context=load_context(prep,plan_path,args.plan_sha256)
    plan,controller,_,assets,manifest,registry,selections,*_=context
    output.mkdir(parents=True)
    shared.PLAN_SHA=args.plan_sha256
    recovery=plan['recovery']
    shared.verify(args.calibration_archive,recovery['calibration_archive'])
    acceptance=prep/'calibration_local_acceptance.json';shared.verify(acceptance,recovery['independent_acceptance'])
    validator=shared.load_module('original_calibration_validator',prep/'validate_calibration_local.py')
    accepted_calibration=validator.validate(prep,args.calibration_receipt,args.calibration_archive,output/'calibration_validation.json',args.old_prep)
    require(accepted_calibration==shared.read_json(acceptance),'Original independent calibration acceptance changed')
    with shared.checked_zip(args.calibration_archive) as archive:
        raw=archive.read('calibration/calibration_report.json')
    require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==recovery['calibration_report'],'Preserved calibration report changed')
    calibration=json.loads(raw)
    development=shared.read_json(args.development_acceptance)
    shared.verify(args.development_acceptance.parent/'validation.json',development['local_validation'])
    dev_validation=shared.read_json(args.development_acceptance.parent/'validation.json')
    require(dev_validation['status']=='accepted' and dev_validation['physical_cache_complete'] is True,'Development validation incomplete')
    dev_receipt=shared.unpack_phase(args.development_archive,args.development_receipt,'development',plan,output/'development_phase')
    require(development['archive']==dev_receipt['archive'],'Frozen acceptance belongs to another development archive')
    freeze=shared.read_json(args.freeze)
    conditions=frozen_conditions(freeze,development,dev_validation['balanced_scores'],registry,controller,calibration,
                                 shared.identity(args.development_acceptance),args.plan_sha256)
    receipt=shared.unpack_phase(args.phase_archive,args.receipt,'confirmation',plan,output/'phase')
    phase=output/'phase'
    require((phase/Path(plan['confirmation_freeze_path']).name).read_bytes()==args.freeze.read_bytes()
            and (phase/Path(plan['accepted_development_path']).name).read_bytes()==args.development_acceptance.read_bytes(),
            'Archived freeze/development acceptance changed')
    ledger=shared.read_json(phase/'ledger_snapshot.json');controller.ledger_valid(ledger,args.plan_sha256)
    require(ledger['spent_seconds']>=SPENT_FLOOR and ledger['attempts'][-1]['stage']=='confirmation'
            and ledger['attempts'][-1]['status']=='complete','Campaign lost historical spend or confirmation completion')
    remote=PurePosixPath(receipt['archive']['local_path']).with_suffix('')
    require(remote.parent==shared.REMOTE and remote.name.startswith('sensitivity_confirmation_'),'Foreign confirmation output')
    jobs=receipt['state']['jobs'];expected=[]
    for seed in SEEDS:
        for condition in conditions:
            for dataset in DATASETS:
                trial=condition['trial'];root=Path(str(remote))/('seed%d'%seed)/condition['id']/dataset
                expected.append(dict(condition_id=condition['id'],role=condition['role'],trial=trial,dataset=dataset,seed=seed,
                    result_root=str(root),command=controller.command_for(plan,trial,dataset,'confirmation',seed,root,calibration)))
    require(len(jobs)==len(expected)==6*len(conditions),'Confirmation job grid incomplete/duplicated')
    alignment,label_proofs=label_references(context,args.modelnet_labels)
    paired,heads,lion={}, {}, []
    outcomes,raw_rows,case_rows,corruption_rows,timing_rows,caches={},[],[],[],[],{}
    for number,(job,wanted) in enumerate(zip(jobs,expected)):
        require(all(job[k]==value for k,value in wanted.items()) and job['status']=='complete' and len(job['raw_archives'])==4,
                'Unregistered condition/dataset/seed/command')
        require(shared.read_json(phase/('job_%03d_validation.json'%number))==job,'Per-job archived validation differs')
        geometry=None;seen=set()
        for raw_archive in job['raw_archives']:
            relative=PurePosixPath(raw_archive['local_path']).relative_to(remote)
            require('..' not in relative.parts,'Unsafe raw archive reference')
            path=phase.joinpath(*relative.parts);shared.verify(path,raw_archive)
            require(raw_archive['drive_path'].startswith(shared.DRIVE),'Raw Drive archive outside owned root')
            config,scores,rows,traces,costs=shared.inspect_raw(path,job,'confirmation',context,calibration,False,alignment,paired,heads,lion)
            head=config['backbone'];require(head==raw_archive['backbone'] and head not in seen,'Duplicate/foreign frozen head');seen.add(head)
            require(path.name==config['run_id']+'.zip' and traces==job['traces'],'Raw trace/run identity differs')
            require(geometry is None or geometry==traces,'Four frozen heads saw different geometry');geometry=traces
            measured=raw_outcomes(path,job['trial']['method']=='original3dd',context[-1],context[-2])
            for policy,values in measured.items():
                require(values.shape==(15,140),'Confirmation selected coverage incomplete')
                key=job['dataset'],head,job['condition_id'],policy,job['seed']
                require(key not in outcomes,'Duplicate condition/policy/seed outcomes');outcomes[key]=values
            for row in scores:case_rows.append(dict(row,condition_id=job['condition_id'],seed=job['seed']))
            for row in rows:corruption_rows.append(dict(row,condition_id=job['condition_id'],seed=job['seed']))
            for row in costs:timing_rows.append(dict(row,condition_id=job['condition_id'],seed=job['seed']))
            raw_rows.append(dict(raw_archive,condition_id=job['condition_id'],seed=job['seed'],run_id=config['run_id'],source_manifest_sha256=config['asset_manifest_sha256'],
                                 canonical_path=shared.copy_canonical(path,prep.parents[2],config)))
        require(seen==set(HEADS),'Missing frozen head')
        for cached in job['cache_records']:
            key=cached['local_path'];require(key not in caches or caches[key]==cached,'Cache receipt changed across conditions');caches[key]=cached
    require(len(raw_rows)==24*len(conditions),'Wrong final raw archive count')
    require(lion==[calibration['lion_state_before']] and ledger['lion_state']==lion[0], 'LION changed from accepted calibration/campaign ledger')
    validate_classifier_contracts(heads, ledger)
    physical=validate_physical_caches(list(caches.values()),paired,args.cache_root)
    shared.write_json(output/'cache_validation.json',physical)
    names=[c['id'] for c in conditions];comparators=[c for c in names if c!='spectral-candidate']
    paired_cases,contrasts=paired_tables(outcomes,context[-1],comparators)
    corruption_deltas=[]
    for comparator in comparators:
        policy='terminal' if comparator=='original3dd' else 'probability_mean'
        for seed in SEEDS:
            for dataset in DATASETS:
                for head in HEADS:
                    left=outcomes[(dataset,head,'spectral-candidate','probability_mean',seed)]
                    right=outcomes[(dataset,head,comparator,policy,seed)]
                    for i,corruption in enumerate(context[-1]):
                        difference=100.*float((left[i].astype(float)-right[i].astype(float)).mean())
                        corruption_deltas.append(dict(comparator=comparator,seed=seed,dataset=dataset,backbone=head,corruption=corruption,
                            delta_pp=difference,corrected_events=int((left[i]&~right[i]).sum()),harmed_events=int((~left[i]&right[i]).sum()),
                            sign='positive' if difference>0 else 'negative' if difference<0 else 'tied'))
    for name,rows in [('case_policy.csv',case_rows),('corruption_policy.csv',corruption_rows),
                      ('balanced_policy.csv',balanced_table(outcomes,names,context[-1])),('paired_cases.csv',paired_cases),
                      ('paired_contrasts.csv',contrasts),('corruption_contrasts.csv',corruption_deltas),('costs.csv',cost_table(timing_rows))]:
        shared.write_csv(output/name,rows)
    shared.write_json(output/'raw_archives.json',raw_rows)
    complete=all(r['local_verified'] for r in physical)
    validation=dict(status='accepted' if complete else 'validated_pending_physical_cache',stage='confirmation',phase_plan_sha256=args.plan_sha256,
        session=plan['session'],manifest_sha256=plan['manifest_sha256'],source_snapshot_sha256=plan['source_snapshot_sha256'],archive=receipt['archive'],
        frozen_selection=shared.identity(args.freeze),development_acceptance=shared.identity(args.development_acceptance),
        calibration_acceptance=shared.identity(output/'calibration_validation.json'),conditions=names,jobs=len(jobs),raw_archives=len(raw_rows),
        physical_cache_files=len(physical),physically_verified_cache_files=sum(r['local_verified'] for r in physical),physical_cache_complete=complete,
        n_original_object_clusters=280,full_benchmark=False,prior_benchmark_exposure=True,campaign_spent_seconds=ledger['spent_seconds'],
        label_references=label_proofs,
        selection_policy='probability_mean',primary_contrasts=[r for r in contrasts if r['scope']=='all15' and r['seed'] is None],
        spectral_family=conditions[0]['trial']['family'],
        limits='Conditional object-cluster intervals across3fixedseeds; no equivalence/universal superiority. Same-host C+S-minus-C isolates spectral removal at selectedrates; S-only-minus-C replaces objective. Original comparison changes host/batching/readout. Macro14/otherreadouts/contrasts are unadjusted diagnostics. Shared reconstruction countedonce; classifiercosts summed, peakmax. Replay/instrumented costs are not deploymentlatency. Calibration retains originalr1provenance.')
    shared.write_json(output/'validation.json',validation)
    accepted=dict(status=validation['status'],stage='confirmation',phase_plan_sha256=args.plan_sha256,archive=receipt['archive'],
                  local_validation=shared.identity(output/'validation.json'),physical_cache_complete=complete,
                  n_original_object_clusters=280,frozen_selection=validation['frozen_selection'])
    shared.write_json(output/'accepted_sensitivity_confirmation.json',accepted)
    print(json.dumps(dict(status=validation['status'],output=str(output),raw_archives=len(raw_rows),physical_cache_complete=complete)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prep','old-prep','plan','phase-archive','receipt','calibration-archive','calibration-receipt',
                 'development-archive','development-receipt','development-acceptance','freeze','modelnet-labels','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--cache-root',type=Path,action='append',default=[])
    run(parser.parse_args())
