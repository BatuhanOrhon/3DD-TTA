"""Post-failure exploratory amendment; frozen scientific workers remain unchanged.

`inspect` is CPU only. `run` requires the separately sealed explicit amendment
acknowledgment and original duration approval. An interrupted continuation stops
for reconciliation; no attempted slot is ever retried.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as a
import campaign as c
import operations as op
import numpy as np

LABEL = 'post-failure exploratory protocol amendment'
NONFINITE = 'ValueError: Classifier input contains nonfinite XYZ coordinates'
SOURCES = [Path(__file__).resolve(), Path(__file__).with_name('test_continue_campaign.py')]


def source_identities() -> dict:
    return {p.name: c.identity(p) for p in SOURCES}


def sealed_raw(slot: Path) -> Path:
    manifests = list((slot / 'raw').rglob('bundle_manifest.json'))
    c.require(len(manifests) == 1, 'Exactly one sealed worker root required')
    raw = Path(str(manifests[0].parent) + '.zip')
    c.require(raw.is_file(), 'Sealed worker ZIP missing')
    return raw


def numerical_failure(slot: Path, plan: dict, job: dict, root: Path,
                      accepted_reports: list) -> dict:
    """Verify the frozen guidance-path clean-probe XYZ rejection; never score a prefix."""
    c.require(not (slot / 'acceptance.json').exists(), 'Failed slot also has acceptance')
    c.require(c.read(slot / 'job.json') == job, 'Failed job identity changed')
    failure = c.read(slot / 'failure.json')
    c.require(failure.get('error') == "ValueError('Worker failed: 1')", 'Not the observed worker exit')
    raw = sealed_raw(slot)
    bundle = a.Bundle(raw)
    try:
        cfg = bundle.json('config.json')
        a.subset(plan['binding']['expected_config'], cfg)
        a.subset(dict(normal_reverse_steps=job['config']['normal_reverse_steps'], seed=job['seed'],
                      method='3dd_chamfer_depth_study', dataset='modelnet40_c', backbone='pointmae',
                      batch_size=32, severity=5, background_reverse_steps=35, technical_smoke=False,
                      guidance_rate_mapping=dict(gamma='local_latent', eta='conditioning_style')),
                 cfg)
        a.subset(dict(total=100, scd_normalize=False, **{k: job['config'][k] for k in ('p', 'eta', 'gamma')}),
                 cfg['guidance_protocol'])
        a.subset(dict(temperature=1., style='original encoded shape_latent for every clean and terminal decode',
                      labels_used=False, early_stopping=False), cfg['trajectory_selection'])
        c.require(cfg.get('status') == 'failed' and cfg.get('worker_returncode') == 1 and cfg.get('corruptions') == list(a.CORRUPTIONS),
                  'Not an exact failed all15 worker')
        trace = cfg.get('failure', '')
        console = (slot / 'console.log').read_text()
        frames = [('scripts/eval_chamfer_depth.py', 'reconstruct_geometry'),
                  ('tta.py', 'tta_reconstruct'), ('shared_trajectory.py', 'clean'),
                  ('scripts/eval_multibackbone_guidance.py', 'classify_decoded'),
                  ('scripts/eval_multibackbone_guidance.py', 'classify_geometry'),
                  ('classifier_backbones.py', 'forward')]
        positions = []
        for filename, function in frames:
            matches = [trace.index(line) for line in trace.splitlines()
                       if ('/' + filename + '"') in line and line.endswith(', in ' + function)]
            c.require(len(matches) == 1, 'Nonfinite failure has different frozen call path')
            positions.append(matches[0])
        c.require(positions == sorted(positions) and trace.rstrip().endswith(NONFINITE) and
                  trace.count('Traceback (most recent call last):') == 1 and
                  console.count('Traceback (most recent call last):') == 1 and
                  trace.split('\n', 1)[1].strip() in console, 'Failure trace/console mismatch or additional exception')
        c.require(not any(k in cfg for k in ('lion_state_audit_error', 'classifier_state_audit_error', 'parent_failure')),
                  'Worker state/source audit failed')
        c.require(c.is_sha(cfg.get('classifier_state_before_sha256')) and
                  cfg['classifier_state_before_sha256'] == cfg.get('classifier_state_after_sha256') and
                  cfg.get('lion_state_before') and cfg['lion_state_before'] == cfg.get('lion_state_after') and
                  all(c.is_sha(v) for v in cfg['lion_state_before'].values()), 'Failed worker model state changed')
        selection = plan['selections'][job['split']]['manifest']
        ids = selection['indices']
        selection_bytes = bundle.read('selection_manifest.json')
        c.require(bundle.json('selection_manifest.json') == selection and cfg.get('selected_original_ids') == ids and
                  cfg.get('evaluation_n_examples') == len(ids) and
                  cfg.get('selection_manifest_identity') == dict(bytes=len(selection_bytes),
                      sha256=hashlib.sha256(selection_bytes).hexdigest()), 'Failed worker selected IDs changed')
        a.subset(plan['binding']['required_source_files'], cfg.get('executed_source_files', {}))
        c.require(cfg.get('executed_source_sha256') == hashlib.sha256(json.dumps(cfg['executed_source_files'],
                  sort_keys=True, separators=(',', ':')).encode()).hexdigest(), 'Executed source map changed')
        for name, expected in plan['binding']['required_source_files'].items():
            if name.startswith('repo/'):
                data = bundle.read('source_snapshot/' + name[len('repo/'):])
            elif name.startswith('assets/source_snapshot/'):
                archive = bundle.read('source_snapshot.zip')
                c.require(hashlib.sha256(archive).hexdigest() == cfg['source_snapshot_sha256'], 'Source archive changed')
                with zipfile.ZipFile(io.BytesIO(archive)) as snapshot:
                    member = name[len('assets/'):]
                    c.require(snapshot.namelist().count(member) == 1 and snapshot.testzip() is None,
                              'Source member missing/duplicate/CRC invalid')
                    data = snapshot.read(member)
            else:
                continue
            c.require(expected == dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
                      'Preserved source bytes changed: ' + name)
        proofs = {p['relative_path']: p for r in accepted_reports for p in r['cache_proofs']
                  if r['stage'] == ('confirmation' if job['split'] == 'heldout' else 'development') and r['seed'] == job['seed']}
        completed = cfg.get('completed_batches', {})
        counts = [completed.get(corruption, 0) for corruption in a.CORRUPTIONS]
        total = (len(ids) + 31) // 32
        c.require(all(type(n) is int and 0 <= n <= total for n in counts) and
                  counts != [total] * 15, 'Invalid failed completion counts')
        active = next(i for i, n in enumerate(counts) if n < total)
        c.require(all(n == 0 for n in counts[active + 1:]), 'Non-prefix failed completion')
        used, cache_proofs = set(), []
        for index, corruption in enumerate(a.CORRUPTIONS):
            for batch in range(counts[index]):
                prefix = 'batches/%s/batch%04d/' % (corruption, batch)
                used.add(prefix + 'predictions.npz')
                with np.load(io.BytesIO(bundle.read(prefix + 'predictions.npz')), allow_pickle=False) as data:
                    batch_ids = ids[batch * 32:(batch + 1) * 32]
                    depth = 35 if corruption == 'background' else job['config']['normal_reverse_steps']
                    steps = a.clean_steps(depth)
                    names = ['clean_%02d' % s for s in steps] + ['terminal']
                    logits = data['trajectory_logits']
                    c.require(data['index'].tolist() == batch_ids and logits.shape == (len(batch_ids), len(names), 40)
                              and np.isfinite(logits).all() and data['candidate_names'].tolist() == names and
                              data['candidate_loop_counts'].tolist() == steps + [depth], 'Invalid completed-prefix predictions')
                    c.require(np.array_equal(data['logits'], logits[:, -1]) and
                              np.array_equal(data['prediction'], logits[:, -1].argmax(1)), 'Prefix terminal mismatch')
                    c.require(data['index'].dtype.kind in 'iu' and data['label'].shape == (len(batch_ids),) and
                              data['label'].dtype.kind in 'iu' and ((data['label'] >= 0) & (data['label'] < 40)).all(),
                              'Invalid prefix ID/label types or ranges')
                    policies = a.policies(logits, len(names) - 1)
                    for key in a.POLICIES + ('selected_candidate', 'entropy', 'probabilities'):
                        c.require(np.allclose(data[key], policies[key], rtol=1e-10, atol=1e-12), 'Prefix policy mismatch')
                diagnostic = bundle.json(prefix + 'diagnostics.json')
                trajectory = diagnostic['trajectory']
                scheduler = diagnostic['scheduler']
                c.require(scheduler.get('active_timesteps') == list(range((depth - 1) * 10, -1, -10)) and
                          scheduler.get('reverse_steps') == depth and scheduler.get('initial_timestep') == (depth - 1) * 10 and
                          scheduler.get('config', {}).get('set_alpha_to_one') is True and scheduler.get('final_alpha_cumprod') == 1.,
                          'Failed-prefix actual scheduler changed')
                c.require(trajectory.get('complete') is True and trajectory.get('horizon') == depth and
                          trajectory.get('candidate_names') == names and
                          trajectory.get('candidate_loop_counts') == steps + [depth] and
                          [p['loop_count'] for p in trajectory['probes']] == steps and diagnostic.get('rng_unchanged') is True,
                          'Incomplete prefix probe/RNG diagnostics')
                c.require(trajectory.get('final_hashes', {}).get('style') == trajectory['initial_hashes']['shape_latent'],
                          'Failed-prefix terminal style changed')
                for probe in trajectory['probes']:
                    c.require(all(probe.get(k) is True for k in ('rng_unchanged', 'buffers_unchanged', 'live_inputs_unchanged'))
                              and probe.get('timestep') == (depth - probe['loop_count']) * 10
                              and c.is_sha(probe.get('rng_before_sha256')) and probe['rng_before_sha256'] == probe.get('rng_after_sha256')
                              and probe.get('buffers_before') and probe['buffers_before'] == probe.get('buffers_after') and
                              probe.get('style_sha256') == trajectory['initial_hashes']['shape_latent'], 'Prefix probe purity failed')
                a.subset({k: cfg[k] for k in ('executed_source_sha256', 'source_snapshot_sha256',
                                               'asset_manifest_sha256', 'checkpoint_load')}, diagnostic)
                cached = diagnostic['paired_state']
                relative = ('paired/seed%d/modelnet40_c/%s/%s' % (job['seed'], corruption, Path(cached['path']).name))
                c.require(Path(cached['path']).parent == Path(cfg['paired_state_root']) / 'modelnet40_c' / corruption,
                          'Failed raw cache namespace changed')
                a.verify_cache(root / relative, cached, trajectory['initial_hashes'], len(batch_ids))
                c.require(proofs.get(relative) == dict(relative_path=relative, **{k: cached[k] for k in ('bytes', 'sha256')}),
                          'Failed-prefix cache lacks accepted baseline proof')
                cache_proofs.append(proofs[relative])
        c.require(used == {n for n in bundle.names if n.startswith('batches/') and n.endswith('/predictions.npz')},
                  'Undeclared failed-prefix shard')
        corruption, batch = a.CORRUPTIONS[active], counts[active]
        batch_ids = ids[batch * 32:(batch + 1) * 32]
        filename = hashlib.sha256(np.asarray(batch_ids, dtype=np.int64).tobytes()).hexdigest() + '.npz'
        relative = 'paired/seed%d/modelnet40_c/%s/%s' % (job['seed'], corruption, filename)
        c.require(relative in proofs and c.identity(root / relative) == {k: proofs[relative][k] for k in ('bytes', 'sha256')},
                  'Failed-batch preparation lacks matching accepted physical baseline cache')
        return dict(status='infeasible', accuracy_pct=None, config_id=job['config_id'], seed=job['seed'],
                    stage='confirmation' if job['split'] == 'heldout' else 'development',
                    readout='probability_mean', plan_sha256=c.digest(plan), bundle_sha256=c.identity(raw)['sha256'],
                    reason=NONFINITE, classification=LABEL, failed_corruption=corruption, failed_batch=batch,
                    failed_probe_loop_count=None, failed_probe_rng_equality=None, failed_probe_buffer_equality=None,
                    evidence_limit='Frozen guard and traceback prove guidance-path predicted-clean classifier-input nonfinite rejection. Decoder/latent root cause is not isolated; failing probe arrays and after-probe purity were not captured.',
                    console_identity=c.identity(slot / 'console.log'), cache_proofs=cache_proofs + [proofs[relative]])
    finally:
        bundle.close()


def feasible_rank(configs: list, scores: list) -> list:
    lookup = {r['config_id']: r for r in scores}
    c.require(len(lookup) == len(scores), 'Duplicate configuration endpoint')
    feasible = []
    for value in configs:
        row = lookup.get(c.config_id(value), {})
        c.require(row.get('stage') == 'development' and row.get('seed') == 0 and
                  row.get('readout') == 'probability_mean' and c.is_sha(row.get('bundle_sha256')),
                  'Every original grid point must have a verified attempted endpoint')
        if row.get('status') == 'infeasible':
            c.require(row.get('accuracy_pct') is None and row.get('classification') == LABEL and row.get('reason') == NONFINITE,
                      'Invalid infeasible endpoint')
        else:
            c.require(row.get('status') == 'accepted' and isinstance(row.get('accuracy_pct'), (int, float)) and
                      math.isfinite(row['accuracy_pct']), 'Endpoint unavailable')
            feasible.append(value)
    c.require(feasible, 'No feasible finite endpoint remains')
    return c.ranked(feasible, [lookup[c.config_id(v)] for v in feasible])


def amended_stage(plan_path: Path, scores: list, phase: str) -> dict:
    plan = c.read(plan_path)
    c.validate_plan(plan)
    c.require(all(r.get('plan_sha256') == c.identity(plan_path)['sha256'] for r in scores), 'Foreign score plan')
    best_a = c.ranked(plan['stage_a'], scores)
    depths = [v['normal_reverse_steps'] for v in best_a[:2]]
    grid = [c.config(n, e, g) for n in depths for e in c.RATES for g in c.RATES]
    prior = list(plan['stage_a'])
    if phase == 'rates':
        c.exact_score_scope(scores, prior)
        values, seeds = grid, [0]
    else:
        best_rates = [feasible_rank([v for v in grid if v['normal_reverse_steps'] == n], scores)[0] for n in depths]
        fractions = [c.config(v['normal_reverse_steps'], v['eta'], v['gamma'], p)
                     for v in best_rates for p in plan['p_values']]
        prior += grid
        if phase == 'fraction':
            c.exact_score_scope(scores, prior)
            values, seeds = fractions, [0]
        else:
            c.require(phase == 'confirmation', 'Unknown amendment phase')
            c.exact_score_scope(scores, prior + fractions)
            values, seeds = [c.config(5), best_a[0], feasible_rank(fractions, scores)[0]], [0, 1, 2]
    jobs, reused, seen = [], [], set()
    for value in values:
        key = c.config_id(value)
        if key in seen:
            continue
        seen.add(key)
        if phase != 'confirmation' and key in {c.config_id(v) for v in prior}:
            reused.append(key)
            continue
        jobs += [dict(config=value, config_id=key, seed=seed, split='heldout' if phase == 'confirmation' else 'development',
                      technical_smoke=False, slot_id=phase + '-' + key + '-s' + str(seed)) for seed in seeds]
    return dict(schema_version=1, phase=phase, plan_sha256=c.identity(plan_path)['sha256'],
                score_input_sha256=None, frozen_conditions=list({c.config_id(v): v for v in values}.values()) if phase == 'confirmation' else None,
                condition_roles=[dict(role=role, config_id=c.config_id(v)) for role, v in zip(plan['confirmation_roles'], values)]
                    if phase == 'confirmation' else None, jobs=jobs, reused_config_ids=reused)


def collect_scores(plan_path: Path, root: Path, output: Path) -> list:
    a.analyze_campaign(plan_path, root, output, 10000)
    analysis = c.read(output / 'analysis.json')
    scores = c.read(output / 'scores.json')
    for unavailable in analysis['unavailable_endpoints']:
        slot = root / 'slots' / unavailable['slot_id']
        job = c.read(slot / 'job.json')
        report = numerical_failure(slot, c.read(plan_path), job, root, analysis['accepted_reports'])
        existing = slot / 'numerical_infeasibility.json'
        if existing.exists():
            c.require(c.read(existing) == report, 'Immutable infeasibility receipt changed')
        else:
            c.write_new(existing, report)
        if report['stage'] == 'development':
            scores.append(report)
    c.write_new(output / 'amended_scores.json', scores)
    return scores


def validate_amendment(plan_path: Path, root: Path, amendment: dict, proposal_path: Path,
                       acknowledgment_path: Path | None = None) -> dict:
    plan = c.check_approval(plan_path, c.read(root / 'approval.json'))
    acknowledgment_path = acknowledgment_path or proposal_path.with_name('continuation_user_acknowledgment.json')
    acknowledgment, proposal = c.read(acknowledgment_path), c.read(proposal_path)
    approval_source = proposal_path.with_name('original_gpu_approval_source.json')
    c.require(c.read(approval_source) == c.read(root / 'approval.json'),
              'Original approval source differs from registered approval contents')
    c.require(c.read(root / 'plan.json') == plan, 'Original campaign plan changed')
    rows, spent = c.ledger_events(root / 'wall_ledger.jsonl')
    c.require(all(r.get('plan_sha256') == c.identity(plan_path)['sha256'] for r in rows), 'Foreign ledger plan')
    c.require(amendment.get('approved') is True and amendment.get('user_statement') == 'Bu protokol değişikliğiyle devam et (önerilen)'
              and amendment.get('approved_at_utc') and amendment.get('classification') == LABEL,
              'Explicit post-failure amendment acknowledgment missing')
    c.require(acknowledgment.get('approved') is True and acknowledgment.get('user_statement') == amendment['user_statement'] and
              acknowledgment.get('recorded_at_utc') == amendment['approved_at_utc'] and
              acknowledgment.get('original_plan_sha256') == c.identity(plan_path)['sha256'] and
              acknowledgment.get('original_gpu_approval_sha256') == c.identity(approval_source)['sha256'] and
              acknowledgment.get('approved_proposal_sha256') == c.identity(proposal_path)['sha256'] and
              acknowledgment.get('cumulative_wall_cap_seconds') == plan['duration']['cumulative_wall_cap_seconds'] and
              acknowledgment.get('failed_slot_must_not_repeat') == proposal.get('failed_slot_preserved') and
              acknowledgment.get('remaining_rate_jobs') == len(proposal.get('remaining_rate_jobs', [])) == 10 and
              acknowledgment.get('max_remaining_gpu_attempts') == proposal.get('max_remaining_gpu_attempts') == 25 and
              proposal.get('original_plan_sha256') == c.identity(plan_path)['sha256'] and
              proposal.get('research_budget_seconds') == plan['duration']['cumulative_wall_cap_seconds'],
              'Actual user acknowledgment/proposal/original GPU approval mismatch')
    c.require(amendment.get('original_plan_sha256') == c.identity(plan_path)['sha256'] and
              amendment.get('source_snapshot_sha256') == plan['binding']['expected_config']['source_snapshot_sha256'] and
              amendment.get('cumulative_wall_cap_seconds') == plan['duration']['cumulative_wall_cap_seconds'] and
              amendment.get('proposal_identity') == c.identity(proposal_path) and
              amendment.get('acknowledgment_identity') == c.identity(acknowledgment_path) and
              amendment.get('original_gpu_approval_identity') == c.identity(root / 'approval.json') and
              amendment.get('original_gpu_approval_source_identity') == c.identity(approval_source) and
              amendment.get('continuation_sources') == source_identities() and
              amendment.get('base_ledger_identity') == c.identity(root / 'wall_ledger.jsonl'),
              'Amendment source/proposal/ledger/cap binding mismatch')
    charge = amendment.get('recovery_charge_seconds', 0.)
    c.require(isinstance(charge, (int, float)) and math.isfinite(charge) and charge >= 0,
              'Invalid recovery charge')
    c.require(spent + charge < plan['duration']['cumulative_wall_cap_seconds'], 'Recovery charge exhausts cap')
    return plan


def run(plan_path: Path, root: Path, amendment_path: Path, proposal_path: Path,
        acknowledgment_path: Path | None = None) -> None:
    amendment = c.read(amendment_path)
    acknowledgment_path = acknowledgment_path or proposal_path.with_name('continuation_user_acknowledgment.json')
    plan = validate_amendment(plan_path, root, amendment, proposal_path, acknowledgment_path)
    c.require(root.resolve() == Path(plan['remote_root']).resolve(), 'Execution must use the original campaign root')
    op.verify_preflight(plan)
    lock = root / 'controller.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, c.encoded(dict(pid=os.getpid(), amendment_sha256=c.digest(amendment))))
    os.close(descriptor)
    ledger = root / 'wall_ledger.jsonl'
    rows, spent = c.ledger_events(ledger)
    sha = c.identity(plan_path)['sha256']
    started = None
    phase = 'continuation_recovery'
    reserve = plan['duration']['final_ledger_sealing_reserve_seconds']
    try:
        for prerequisite in ('smoke', 'depth'):
            registered = c.read(root / ('registered_' + prerequisite + '.json'))
            c.require(registered == c.stage(plan_path, None, prerequisite), 'Original prerequisite registration changed')
            for job in registered['jobs']:
                slot = root / 'slots' / job['slot_id']
                c.require(c.read(slot / 'job.json') == job and not (slot / 'failure.json').exists() and
                          c.read(slot / 'acceptance.json').get('status') == 'accepted' and
                          c.read(slot / 'acceptance.json').get('plan_sha256') == sha,
                          'All original accepted technical-smoke/depth endpoints required')
        c.write_new(root / 'registered_continuation_amendment.json', amendment)
        c.write_new(root / 'registered_continuation_sources.json', source_identities())
        c.write_new(root / 'registered_continuation_proposal.json', c.read(proposal_path))
        c.write_new(root / 'registered_continuation_user_acknowledgment.json', c.read(acknowledgment_path))
        op.copy_verified(proposal_path.with_name('original_gpu_approval_source.json'),
                         root / 'registered_continuation_original_gpu_approval_source.json')
        # Existing ledger accounting includes sealing_overrun charges; use it unchanged.
        if amendment.get('recovery_charge_seconds', 0):
            c.append_event(ledger, rows, event='sealing_overrun', plan_sha256=sha,
                           elapsed_seconds=amendment['recovery_charge_seconds'], reason='explicit amended CPU recovery charge')
        for phase in ('rates', 'fraction', 'confirmation'):
            rows, spent = c.ledger_events(ledger)
            c.require(spent + reserve < plan['duration']['cumulative_wall_cap_seconds'],
                      'Cumulative wall cap exhausted before next CPU phase')
            started = time.monotonic()
            analysis_dir = root / ('continuation_analysis_' + phase)
            try:
                scores = collect_scores(plan_path, root, analysis_dir)
            finally:
                c.append_event(ledger, rows, event='phase_started', phase='continuation_' + phase, plan_sha256=sha)
            development = scores
            if phase == 'rates':
                stage_a_ids = {c.config_id(v) for v in plan['stage_a']}
                stage = amended_stage(plan_path, [r for r in development if r['config_id'] in stage_a_ids], phase)
                original = c.read(root / 'registered_rates.json')
                c.require({k: v for k, v in original.items() if k != 'score_input_sha256'} ==
                          {k: v for k, v in stage.items() if k != 'score_input_sha256'}, 'Original exact rates stage changed')
            else:
                stage = amended_stage(plan_path, development, phase)
            allowed = {j['slot_id'] for p in ('smoke', 'depth') for j in c.read(root / ('registered_' + p + '.json'))['jobs']}
            for previous in ('rates', 'fraction', 'confirmation'):
                registered = root / ('registered_' + previous + '.json')
                if registered.exists():
                    allowed.update(j['slot_id'] for j in c.read(registered)['jobs'])
            allowed.update(j['slot_id'] for j in stage['jobs'])
            c.require({p.name for p in (root / 'slots').iterdir()} <= allowed, 'Unregistered slot exists')
            remaining_jobs = []
            for job in stage['jobs']:
                slot = root / 'slots' / job['slot_id']
                if slot.exists():
                    c.require(phase == 'rates' and c.read(slot / 'job.json') == job and
                              any(r['config_id'] == job['config_id'] for r in scores), 'Already attempted slot; no retry')
                else:
                    remaining_jobs.append(job)
            if phase == 'rates':
                c.require(remaining_jobs == c.read(proposal_path)['remaining_rate_jobs'], 'Remaining rates differ from approved exact ten')
            else:
                stage['score_input_sha256'] = c.identity(analysis_dir / 'amended_scores.json')['sha256']
                c.write_new(root / ('registered_' + phase + '.json'), stage)
            c.write_new(root / ('registered_continuation_' + phase + '.json'),
                        dict(classification=LABEL, amendment_sha256=c.digest(amendment), stage=stage,
                             remaining_jobs=remaining_jobs, scores_identity=c.identity(analysis_dir / 'amended_scores.json')))
            c.write_new(root / ('registered_continuation_' + phase + '_scores.json'), scores)
            op.persist_checkpoint(root, plan['drive_root'], sha, len(rows))
            accepted_reports = c.read(analysis_dir / 'analysis.json')['accepted_reports']
            for job in remaining_jobs:
                # Keep one acceptance allowance plus phase sealing for final CPU analysis.
                final_cpu_reserve = (reserve + plan['duration']['per_job_acceptance_copy_reserve_seconds']) if phase == 'confirmation' else 0
                remaining = plan['duration']['cumulative_wall_cap_seconds'] - spent - (time.monotonic() - started) - reserve - plan['duration']['per_job_acceptance_copy_reserve_seconds'] - final_cpu_reserve
                c.require(remaining > 0, 'Cumulative wall cap exhausted; stop before next worker')
                slot = root / 'slots' / job['slot_id']
                slot.mkdir()
                c.write_new(slot / 'job.json', job)
                manifest = slot / 'selection_manifest.json'
                c.write_new(manifest, plan['selections'][job['split']]['manifest'])
                command = c.worker_command(plan, job, slot / 'raw', manifest, root / 'paired')
                c.write_new(slot / 'command.json', command)
                c.append_event(ledger, rows, event='job_started', slot_id=job['slot_id'], plan_sha256=sha)
                try:
                    returncode = op.run_worker(command, plan['binding']['repo_root'], slot / 'console.log', remaining, job['slot_id'])
                    c.require(returncode == 0, 'Worker failed: %s' % returncode)
                    report = a.analyze_bundle(sealed_raw(slot), plan, job, cache_root=root)
                    c.write_new(slot / 'acceptance.json', report)
                    c.append_event(ledger, rows, event='job_accepted', slot_id=job['slot_id'], bundle_sha256=report['bundle_sha256'], plan_sha256=sha)
                    published = op.persist_checkpoint(root, plan['drive_root'], sha, len(rows), slot, report['cache_proofs'])
                    accepted_reports.append(report)
                except BaseException as error:
                    if not (slot / 'failure.json').exists():
                        c.write_new(slot / 'failure.json', dict(status='unavailable', error=repr(error)))
                    c.append_event(ledger, rows, event='job_failed', slot_id=job['slot_id'], error=repr(error), plan_sha256=sha)
                    published = op.persist_checkpoint(root, plan['drive_root'], sha, len(rows), slot, preserve_raw=True)
                    c.append_event(ledger, rows, event='drive_snapshot_verified', snapshot=published, plan_sha256=sha)
                    # Archive/persistence/acceptance failures cannot become numerical failures.
                    c.require(type(error) is ValueError and str(error) == 'Worker failed: 1', 'Non-numerical failure; stop')
                    infeasible = numerical_failure(slot, plan, job, root, accepted_reports)
                    c.write_new(slot / 'numerical_infeasibility.json', infeasible)
                    published = op.persist_checkpoint(root, plan['drive_root'], sha, len(rows), slot,
                                                      infeasible['cache_proofs'], preserve_raw=True)
                c.append_event(ledger, rows, event='drive_snapshot_verified', snapshot=published, plan_sha256=sha)
                c.require(spent + time.monotonic() - started + reserve <= plan['duration']['cumulative_wall_cap_seconds'],
                          'Cap exceeded during acceptance/copy; stop before next worker')
            phase_started, started = started, None
            finish_phase(root, plan, ledger, rows, phase, phase_started)
        # Frozen analyzer keeps incomplete fixed-seed confirmation unavailable.
        rows, spent = c.ledger_events(ledger)
        c.require(spent + reserve < plan['duration']['cumulative_wall_cap_seconds'],
                  'Cumulative wall cap exhausted before final CPU analysis')
        started = time.monotonic()
        phase = 'final_analysis'
        try:
            collect_scores(plan_path, root, root / 'continuation_analysis_final')
        finally:
            c.append_event(ledger, rows, event='phase_started', phase='continuation_' + phase, plan_sha256=sha)
    finally:
        try:
            if started is not None:
                finish_phase(root, plan, ledger, rows, phase, started)
        finally:
            lock.unlink()


def finish_phase(root: Path, plan: dict, ledger: Path, rows: list, phase: str, started: float) -> None:
    reserve = plan['duration']['final_ledger_sealing_reserve_seconds']
    c.append_event(ledger, rows, event='phase_finished', phase='continuation_' + phase, plan_sha256=c.digest(plan),
                   elapsed_seconds=time.monotonic() - started + reserve, final_ledger_reserved_seconds=reserve)
    sealed_at = time.monotonic()
    op.persist_checkpoint(root, plan['drive_root'], c.digest(plan), len(rows))
    sealing = time.monotonic() - sealed_at
    if sealing > reserve:
        c.append_event(ledger, rows, event='sealing_overrun', plan_sha256=c.digest(plan), elapsed_seconds=sealing - reserve)
        raise RuntimeError('Final ledger persistence exceeded reserve; stop before further work')
    _, charged = c.ledger_events(ledger)
    c.require(charged <= plan['duration']['cumulative_wall_cap_seconds'],
              'Cumulative wall cap exceeded; actual overrun preserved, no further work')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'run'))
    for name in ('plan', 'root', 'proposal', 'acknowledgment'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--amendment', type=Path)
    args = parser.parse_args()
    if args.action == 'inspect':
        rows, spent = c.ledger_events(args.root / 'wall_ledger.jsonl')
        print(json.dumps(dict(classification=LABEL, continuation_sources=source_identities(),
            original_plan_sha256=c.identity(args.plan)['sha256'], proposal_identity=c.identity(args.proposal),
            acknowledgment_identity=c.identity(args.acknowledgment),
            original_gpu_approval_identity=c.identity(args.root / 'approval.json'),
            original_gpu_approval_source_identity=c.identity(args.proposal.with_name('original_gpu_approval_source.json')),
            base_ledger_identity=c.identity(args.root / 'wall_ledger.jsonl'), charged_seconds=spent,
            ledger_events=len(rows)), indent=2))
    else:
        c.require(args.amendment is not None, '--amendment required for run')
        run(args.plan, args.root, args.amendment, args.proposal, args.acknowledgment)


if __name__ == '__main__':
    main()
