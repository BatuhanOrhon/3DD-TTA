"""CPU synthetic contract tests; fixture scores are not research evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as c
import continue_campaign as cont
import operations as op
import test_campaign as fixtures
from test_campaign import plan_path


def ranked_scores(path, values):
    return fixtures.scores(path, values, lambda v: 80 - abs(v['normal_reverse_steps'] - 8)
                           + (v['eta'] == .003) + (v['gamma'] == .003) + (v['p'] == .9))


def infeasible(path, job):
    return dict(config_id=job['config_id'], seed=job['seed'], stage='confirmation' if job['split'] == 'heldout' else 'development',
                readout='probability_mean', plan_sha256=c.identity(path)['sha256'],
                status='infeasible', accuracy_pct=None, bundle_sha256='f' * 64, classification=cont.LABEL,
                reason=cont.NONFINITE, cache_proofs=[])


def test_exact_original_grid_null_infeasible_and_frozen_confirmation(plan_path, tmp_path):
    plan = c.read(plan_path)
    scores = ranked_scores(plan_path, plan['stage_a'])
    score_path = tmp_path / 'scores.json'
    c.write_new(score_path, scores)
    original = c.stage(plan_path, score_path, 'rates')
    amended = cont.amended_stage(plan_path, scores, 'rates')
    assert amended['jobs'] == original['jobs']
    assert len(amended['jobs']) == 16
    assert len(amended['reused_config_ids']) == 2
    for job in amended['jobs']:
        scores += ranked_scores(plan_path, [job['config']])
    failed_job = amended['jobs'][5]
    scores = [r for r in scores if r['config_id'] != failed_job['config_id']] + [infeasible(plan_path, failed_job)]
    fraction = cont.amended_stage(plan_path, scores, 'fraction')
    assert len(fraction['jobs']) == 6
    assert len(fraction['reused_config_ids']) == 2
    assert failed_job['config_id'] not in {j['config_id'] for j in fraction['jobs']}
    with pytest.raises(ValueError, match='Every original grid point'):
        cont.amended_stage(plan_path, scores[:-1], 'fraction')
    broken = copy.deepcopy(scores)
    broken[-1]['accuracy_pct'] = 0
    with pytest.raises(ValueError, match='Invalid infeasible'):
        cont.amended_stage(plan_path, broken, 'fraction')
    for job in fraction['jobs']:
        scores += ranked_scores(plan_path, [job['config']])
    confirmation = cont.amended_stage(plan_path, scores, 'confirmation')
    assert len(confirmation['jobs']) == 9
    assert {j['seed'] for j in confirmation['jobs']} == {0, 1, 2}
    assert all(j['split'] == 'heldout' for j in confirmation['jobs'])
    assert [r['role'] for r in confirmation['condition_roles']] == plan['confirmation_roles']
    assert len(confirmation['frozen_conditions']) == 3


@pytest.mark.parametrize('mutation', ['unavailable', 'nan', 'foreign', 'missing', 'duplicate'])
def test_selection_stops_for_unverified_or_unattempted_endpoint(plan_path, mutation):
    plan = c.read(plan_path)
    scores = ranked_scores(plan_path, plan['stage_a'])
    stage = cont.amended_stage(plan_path, scores, 'rates')
    scores += ranked_scores(plan_path, [j['config'] for j in stage['jobs']])
    if mutation == 'unavailable':
        scores[-1]['status'] = 'unavailable'
    elif mutation == 'nan':
        scores[-1]['accuracy_pct'] = float('nan')
    elif mutation == 'foreign':
        scores[-1]['plan_sha256'] = '0' * 64
    elif mutation == 'missing':
        scores.pop()
    else:
        scores.append(scores[-1])
    with pytest.raises(ValueError):
        cont.amended_stage(plan_path, scores, 'fraction')


def make_root(path, tmp_path):
    root = tmp_path / 'campaign'
    plan = c.read(path)
    plan['remote_root'] = str(root)
    path.write_bytes(c.encoded(plan))
    root.mkdir()
    c.write_new(root / 'plan.json', plan)
    c.write_new(root / 'approval.json', dict(approved=True, user_statement='original approval', approved_at_utc='2026-10-30',
                plan_sha256=c.identity(path)['sha256'], source_snapshot_sha256=plan['binding']['expected_config']['source_snapshot_sha256'],
                cumulative_wall_cap_seconds=10800))
    c.append_event(root / 'wall_ledger.jsonl', [], event='sealing_overrun', elapsed_seconds=3079.138494549,
                   plan_sha256=c.identity(path)['sha256'])
    development = ranked_scores(path, plan['stage_a'])
    for phase in ('smoke', 'depth', 'rates'):
        stage = cont.amended_stage(path, development, phase) if phase == 'rates' else c.stage(path, None, phase)
        c.write_new(root / ('registered_' + phase + '.json'), stage)
        for i, job in enumerate(stage['jobs']):
            if phase == 'rates' and i >= 6:
                break
            slot = root / 'slots' / job['slot_id']
            slot.mkdir(parents=True)
            c.write_new(slot / 'job.json', job)
            if phase == 'rates' and i == 5:
                c.write_new(slot / 'failure.json', dict(status='unavailable', error="ValueError('Worker failed: 1')"))
                c.write_new(slot / 'numerical_infeasibility.json', infeasible(path, job))
            else:
                c.write_new(slot / 'acceptance.json', dict(status='accepted', plan_sha256=c.identity(path)['sha256']))
    proposal_path = tmp_path / 'proposal.json'
    c.write_new(proposal_path, dict(remaining_rate_jobs=c.read(root / 'registered_rates.json')['jobs'][6:],
        failed_slot_preserved=c.read(root / 'registered_rates.json')['jobs'][5]['slot_id'],
        max_remaining_gpu_attempts=25, original_plan_sha256=c.identity(path)['sha256'], research_budget_seconds=10800))
    approval_source = proposal_path.with_name('original_gpu_approval_source.json')
    original = c.read(root / 'approval.json')
    approval_source.write_text(json.dumps({k: original[k] for k in reversed(original)}, indent=2) + '\n')
    assert c.identity(approval_source) != c.identity(root / 'approval.json')
    acknowledgment_path = proposal_path.with_name('continuation_user_acknowledgment.json')
    c.write_new(acknowledgment_path, dict(approved=True, user_statement='Bu protokol değişikliğiyle devam et (önerilen)',
        recorded_at_utc='2026-10-30T15:46:00Z', original_plan_sha256=c.identity(path)['sha256'],
        original_gpu_approval_sha256=c.identity(approval_source)['sha256'], approved_proposal_sha256=c.identity(proposal_path)['sha256'],
        cumulative_wall_cap_seconds=10800, failed_slot_must_not_repeat=c.read(proposal_path)['failed_slot_preserved'],
        remaining_rate_jobs=10, max_remaining_gpu_attempts=25))
    amendment = dict(approved=True, user_statement='Bu protokol değişikliğiyle devam et (önerilen)', approved_at_utc='2026-10-30T15:46:00Z',
        classification=cont.LABEL, original_plan_sha256=c.identity(path)['sha256'],
        source_snapshot_sha256=plan['binding']['expected_config']['source_snapshot_sha256'], cumulative_wall_cap_seconds=10800,
        proposal_identity=c.identity(proposal_path), continuation_sources=cont.source_identities(),
        acknowledgment_identity=c.identity(acknowledgment_path), original_gpu_approval_identity=c.identity(root / 'approval.json'),
        original_gpu_approval_source_identity=c.identity(approval_source),
        base_ledger_identity=c.identity(root / 'wall_ledger.jsonl'), recovery_charge_seconds=12.)
    amendment_path = tmp_path / 'amendment.json'
    c.write_new(amendment_path, amendment)
    return root, amendment_path, proposal_path


def mock_boundaries(monkeypatch, path, root, failure=None):
    attempts, snapshots = [], []
    monkeypatch.setattr(op, 'verify_preflight', lambda plan: None)
    def persist(root, drive, sha, sequence, slot=None, expected_caches=(), preserve_raw=False):
        snapshots.append((sequence, slot, preserve_raw))
        return str(sequence)
    monkeypatch.setattr(op, 'persist_checkpoint', persist)
    def collect(path, root, output):
        output.mkdir()
        scores = []
        for slot in (root / 'slots').iterdir():
            job = c.read(slot / 'job.json')
            if job.get('technical_smoke') or job['split'] == 'heldout':
                continue
            if (slot / 'numerical_infeasibility.json').exists():
                scores.append(c.read(slot / 'numerical_infeasibility.json'))
            else:
                scores += ranked_scores(path, [job['config']])
        c.write_new(output / 'analysis.json', dict(accepted_reports=[]))
        c.write_new(output / 'amended_scores.json', scores)
        return scores
    monkeypatch.setattr(cont, 'collect_scores', collect)
    monkeypatch.setattr(cont, 'sealed_raw', lambda slot: slot / 'fake.zip')
    monkeypatch.setattr(cont.a, 'analyze_bundle', lambda raw, plan, job, cache_root: dict(status='accepted', bundle_sha256='e' * 64, cache_proofs=[]))
    monkeypatch.setattr(cont, 'numerical_failure', lambda slot, plan, job, root, reports: infeasible(path, job))
    def worker(command, cwd, console, timeout, slot_id):
        if slot_id.startswith('confirmation-'):
            freeze = c.read(root / 'registered_confirmation.json')
            assert len(freeze['jobs']) == 9
            assert {j['seed'] for j in freeze['jobs']} == {0, 1, 2}
            assert any(j['slot_id'] == slot_id for j in freeze['jobs'])
            assert any(slot is None for sequence, slot, raw in snapshots)
        assert timeout > 0
        attempts.append(slot_id)
        if failure and len(attempts) == 1:
            if isinstance(failure, BaseException):
                raise failure
            return failure
        return 0
    monkeypatch.setattr(op, 'run_worker', worker)
    return attempts, snapshots


def test_complete_continuation_no_retries_exact25_and_budget_chain(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    original_ledger = (root / 'wall_ledger.jsonl').read_bytes()
    failed_original = (root / 'slots' / c.read(root / 'registered_rates.json')['jobs'][5]['slot_id'] / 'failure.json').read_bytes()
    attempts, snapshots = mock_boundaries(monkeypatch, plan_path, root)
    cont.run(plan_path, root, amendment, proposal)
    assert len(attempts) == len(set(attempts)) == 25
    assert sum(s.startswith('rates-') for s in attempts) == 10
    assert sum(s.startswith('fraction-') for s in attempts) == 6
    assert sum(s.startswith('confirmation-') for s in attempts) == 9
    assert (root / 'wall_ledger.jsonl').read_bytes().startswith(original_ledger)
    rows, spent = c.ledger_events(root / 'wall_ledger.jsonl')
    assert spent >= 3079.138494549 + 12 + 4 * 60
    assert len([r for r in rows if r['event'] == 'phase_finished']) == 4
    assert (root / 'slots' / c.read(root / 'registered_rates.json')['jobs'][5]['slot_id'] / 'failure.json').read_bytes() == failed_original
    assert not (root / 'controller.lock').exists()
    with pytest.raises(ValueError, match='binding mismatch'):
        cont.run(plan_path, root, amendment, proposal)
    assert len(attempts) == 25


@pytest.mark.parametrize('failure', [2, RuntimeError('CUDA out of memory'), OSError('transport'), KeyboardInterrupt()])
def test_first_non_numeric_failure_stops_and_preserves(plan_path, tmp_path, monkeypatch, failure):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    attempts, snapshots = mock_boundaries(monkeypatch, plan_path, root, failure)
    with pytest.raises(ValueError, match='Non-numerical failure'):
        cont.run(plan_path, root, amendment, proposal)
    assert len(attempts) == 1
    slot = root / 'slots' / attempts[0]
    assert (slot / 'failure.json').exists()
    assert not (slot / 'numerical_infeasibility.json').exists()
    assert any(s == slot and raw for sequence, s, raw in snapshots)
    assert not (root / 'registered_confirmation.json').exists()
    c.ledger_events(root / 'wall_ledger.jsonl')


def test_new_numeric_failure_kept_null_and_never_replaced(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    attempts, snapshots = mock_boundaries(monkeypatch, plan_path, root, 1)
    cont.run(plan_path, root, amendment, proposal)
    assert len(attempts) == 25
    receipt = c.read(root / 'slots' / attempts[0] / 'numerical_infeasibility.json')
    assert receipt['status'] == 'infeasible' and receipt['accuracy_pct'] is None
    assert attempts.count(attempts[0]) == 1


def test_confirmation_failure_uses_newly_accepted_same_seed_baseline(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    attempts, _ = mock_boundaries(monkeypatch, plan_path, root)
    normal_worker = op.run_worker
    def worker(command, cwd, console, timeout, slot_id):
        normal_worker(command, cwd, console, timeout, slot_id)
        return 1 if slot_id.startswith('confirmation-') and slot_id.endswith('-s0') and '-n5-' not in slot_id else 0
    monkeypatch.setattr(op, 'run_worker', worker)
    def report(raw, plan, job, cache_root):
        return dict(status='accepted', config_id=job['config_id'], seed=job['seed'],
                    stage='confirmation' if job['split'] == 'heldout' else 'development', bundle_sha256='e' * 64, cache_proofs=[])
    monkeypatch.setattr(cont.a, 'analyze_bundle', report)
    seen = []
    def classify(slot, plan, job, root, reports):
        baseline = next(r for r in reports if r.get('config_id') == c.config_id(c.config(5)) and
                        r.get('stage') == 'confirmation' and r.get('seed') == job['seed'])
        seen.append(baseline)
        return infeasible(plan_path, job)
    monkeypatch.setattr(cont, 'numerical_failure', classify)
    cont.run(plan_path, root, amendment, proposal)
    assert len(attempts) == 25 and len(seen) == 2
    assert all(r['seed'] == 0 for r in seen)


@pytest.mark.parametrize('key', ['approved', 'approved_proposal_sha256', 'original_gpu_approval_sha256',
                                 'original_plan_sha256', 'failed_slot_must_not_repeat', 'remaining_rate_jobs'])
def test_actual_acknowledgment_contents_are_verified(plan_path, tmp_path, key):
    root, amendment_path, proposal = make_root(plan_path, tmp_path)
    ack_path = proposal.with_name('continuation_user_acknowledgment.json')
    ack = c.read(ack_path)
    ack[key] = None
    ack_path.write_bytes(c.encoded(ack))
    amendment = c.read(amendment_path)
    amendment['acknowledgment_identity'] = c.identity(ack_path)
    with pytest.raises(ValueError, match='Actual user acknowledgment'):
        cont.validate_amendment(plan_path, root, amendment, proposal)


def test_final_cpu_overrun_preserves_charge_and_cannot_return_success(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    attempts, _ = mock_boundaries(monkeypatch, plan_path, root)
    collect = cont.collect_scores
    clock = [1000.]
    monkeypatch.setattr(cont.time, 'monotonic', lambda: clock[0])
    def final_overrun(path, root, output):
        result = collect(path, root, output)
        if output.name.endswith('_final'):
            clock[0] += 8000
        return result
    monkeypatch.setattr(cont, 'collect_scores', final_overrun)
    with pytest.raises(ValueError, match='cap exceeded; actual overrun preserved'):
        cont.run(plan_path, root, amendment, proposal)
    _, charged = c.ledger_events(root / 'wall_ledger.jsonl')
    assert charged > 10800 and len(attempts) == 25
    assert not (root / 'controller.lock').exists()


def test_no_final_cpu_phase_when_final_copy_consumes_remaining_allowance(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    attempts, snapshots = mock_boundaries(monkeypatch, plan_path, root)
    clock = [1000.]
    monkeypatch.setattr(cont.time, 'monotonic', lambda: clock[0])
    def persist(root, drive, sha, sequence, slot=None, expected_caches=(), preserve_raw=False):
        snapshots.append((sequence, slot, preserve_raw))
        if slot is not None and len(attempts) == 25:
            # Actual acceptance/copy may overrun its allowance; retain its wall.
            clock[0] = 1000 + 10800 - (3079.138494549 + 12 + 2 * 60) - 60 - 1
        return str(sequence)
    monkeypatch.setattr(op, 'persist_checkpoint', persist)
    with pytest.raises(ValueError, match='cap exhausted before final CPU analysis'):
        cont.run(plan_path, root, amendment, proposal)
    _, charged = c.ledger_events(root / 'wall_ledger.jsonl')
    assert 10798 <= charged <= 10800
    assert not (root / 'continuation_analysis_final').exists()
    assert len(attempts) == 25


@pytest.mark.parametrize('key', ['approved', 'user_statement', 'continuation_sources', 'proposal_identity', 'base_ledger_identity', 'cumulative_wall_cap_seconds'])
def test_amendment_requires_actual_ack_and_all_bindings(plan_path, tmp_path, monkeypatch, key):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    value = c.read(amendment)
    value[key] = None
    attempts, _ = mock_boundaries(monkeypatch, plan_path, root)
    with pytest.raises(ValueError):
        cont.validate_amendment(plan_path, root, value, proposal)
    assert attempts == []


def test_no_next_worker_when_recovery_and_reserves_exhaust_cap(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    value = c.read(amendment)
    value['recovery_charge_seconds'] = 10800 - 3079.138494549 - 200
    amendment.write_bytes(c.encoded(value))
    attempts, _ = mock_boundaries(monkeypatch, plan_path, root)
    with pytest.raises(ValueError, match='cap exhausted'):
        cont.run(plan_path, root, amendment, proposal)
    assert attempts == []
    rows, spent = c.ledger_events(root / 'wall_ledger.jsonl')
    assert spent >= 10660


def test_sealing_overrun_is_charged_and_stops_without_double_finish(plan_path, tmp_path, monkeypatch):
    root, amendment, proposal = make_root(plan_path, tmp_path)
    attempts, _ = mock_boundaries(monkeypatch, plan_path, root)
    clock = [1000.]
    monkeypatch.setattr(cont.time, 'monotonic', lambda: clock[0])
    def persist(root, drive, sha, sequence, slot=None, expected_caches=(), preserve_raw=False):
        last = (root / 'wall_ledger.jsonl').read_text().splitlines()[-1]
        if 'phase_finished' in last:
            clock[0] += 61
        return str(sequence)
    monkeypatch.setattr(op, 'persist_checkpoint', persist)
    with pytest.raises(RuntimeError, match='exceeded reserve'):
        cont.run(plan_path, root, amendment, proposal)
    rows, spent = c.ledger_events(root / 'wall_ledger.jsonl')
    assert len(attempts) == 10
    assert len([r for r in rows if r['event'] == 'phase_finished']) == 1
    assert rows[-1]['event'] == 'sealing_overrun' and rows[-1]['elapsed_seconds'] == 1


def failed_bundle(tmp_path, plan, job):
    raw, folder = fixtures.fixture_bundle(tmp_path, plan, job)
    cfg = c.read(folder / 'config.json')
    frames = [('scripts/eval_chamfer_depth.py', 'reconstruct_geometry'), ('tta.py', 'tta_reconstruct'),
              ('shared_trajectory.py', 'clean'), ('scripts/eval_multibackbone_guidance.py', 'classify_decoded'),
              ('scripts/eval_multibackbone_guidance.py', 'classify_geometry'), ('classifier_backbones.py', 'forward')]
    trace = 'Traceback (most recent call last):\n' + ''.join('  File "/synthetic/%s", line 1, in %s\n' % pair for pair in frames) + cont.NONFINITE + '\n'
    cfg.update(status='failed', worker_returncode=1, failure=trace, completed_batches={'uniform': 4, 'gaussian': 4})
    cfg['executed_source_sha256'] = hashlib.sha256(json.dumps(cfg['executed_source_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (folder / 'config.json').write_bytes(c.encoded(cfg))
    import shutil
    for path in (folder / 'batches').iterdir():
        if path.name not in ('uniform', 'gaussian'):
            shutil.rmtree(path)
    fixtures.seal_fixture(folder)
    slot = tmp_path / 'slot'
    (slot / 'raw').mkdir(parents=True)
    shutil.move(str(folder), slot / 'raw' / folder.name)
    shutil.move(str(raw), slot / 'raw' / raw.name)
    c.write_new(slot / 'job.json', job)
    c.write_new(slot / 'failure.json', dict(status='unavailable', error="ValueError('Worker failed: 1')"))
    (slot / 'console.log').write_text(trace)
    proofs = [dict(relative_path=p.relative_to(tmp_path).as_posix(), **c.identity(p)) for p in (tmp_path / 'paired').rglob('*.npz')]
    return slot, [dict(stage='development', seed=0, cache_proofs=proofs)]


@pytest.mark.parametrize('mutation', [None, 'source', 'model', 'trace', 'console', 'additional_exception', 'probe', 'cache', 'ids'])
def test_raw_numeric_classification_requires_intact_actual_evidence(plan_path, tmp_path, mutation):
    plan = c.read(plan_path)
    value = c.config(8, .03, .003)
    job = dict(config=value, config_id=c.config_id(value), seed=0, split='development', technical_smoke=False, slot_id='fixture')
    slot, reports = failed_bundle(tmp_path, plan, job)
    folder = slot / 'raw/raw_bundle'
    cfg = c.read(folder / 'config.json')
    if mutation == 'source':
        cfg['source_snapshot_sha256'] = '0' * 64
    elif mutation == 'model':
        cfg['classifier_state_after_sha256'] = '0' * 64
    elif mutation == 'trace':
        cfg['failure'] = cfg['failure'].replace('in clean', 'in terminal')
    elif mutation == 'console':
        (slot / 'console.log').write_text('unrelated error')
    elif mutation == 'additional_exception':
        with (slot / 'console.log').open('a') as stream:
            stream.write('Traceback (most recent call last):\nOSError: archive delivery failed\n')
    elif mutation == 'probe':
        p = folder / 'batches/uniform/batch0000/diagnostics.json'
        diagnostic = c.read(p)
        diagnostic['trajectory']['probes'][0]['rng_unchanged'] = False
        p.write_bytes(c.encoded(diagnostic))
    elif mutation == 'cache':
        reports[0]['cache_proofs'] = []
    elif mutation == 'ids':
        cfg['selected_original_ids'] = cfg['selected_original_ids'][1:]
    (folder / 'config.json').write_bytes(c.encoded(cfg))
    fixtures.seal_fixture(folder)
    if mutation:
        with pytest.raises(ValueError):
            cont.numerical_failure(slot, plan, job, tmp_path, reports)
    else:
        report = cont.numerical_failure(slot, plan, job, tmp_path, reports)
        assert report['status'] == 'infeasible' and report['accuracy_pct'] is None
        assert report['failed_corruption'] == 'background' and report['failed_batch'] == 0
        assert report['failed_probe_loop_count'] is None
        assert report['failed_probe_rng_equality'] is None
        assert report['failed_probe_buffer_equality'] is None
