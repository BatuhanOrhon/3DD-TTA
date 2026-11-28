"""Recount the four frozen heads on CPU and compare matched original IDs from verified raw archives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_artifacts import CORRUPTIONS
from scripts.analyze_cloudfixer_multibackbone import read_shard
from scripts.run_cloudfixer_multibackbone import HEADS as THREE_HEADS

MODELS = (*THREE_HEADS, 'pointmae')
N = 2468
BASE_ZIP = 'c392dce804c213a21e97769590e3a036ee80deefb9b0d6379689958530646265'
PLAIN_ZIP = 'bcb64ea0c5c0165e43c4498dfc428af6c1e4804f82889e17d4924b76e38cfe07'
DEV_ZIP = '28baa27f2b80ae32c42fb5b56d569f8bb3dfac4e94fb6c055a2e89d1dabf50e7'
EVIDENCE = ROOT / 'knowledge/evidence'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    with path.open('rb') as src:
        h = hashlib.sha256()
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def npz(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


def checked_logits(values, shape):
    require(values.shape == shape and np.issubdtype(values.dtype, np.floating)
            and np.isfinite(values).all(), f'Invalid logits, expected {shape}')
    return values


def pm_and_terminal(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = checked_logits(logits, (len(logits), 3, 40)).astype(np.float64)
    p = np.exp(x - x.max(axis=-1, keepdims=True))
    return (p / p.sum(axis=-1, keepdims=True)).mean(axis=1).argmax(axis=-1), x[:, -1].argmax(axis=-1)


def record_predictions(target: np.ndarray, ids: np.ndarray, values: np.ndarray) -> None:
    require(ids.ndim == 1 and ids.shape == values.shape and len(ids) > 0
            and ids.dtype.kind in 'iu' and values.dtype.kind in 'iu'
            and ids.min() >= 0 and ids.max() < len(target) and len(np.unique(ids)) == len(ids)
            and np.all(target[ids] == -1) and values.min() >= 0 and values.max() < 40,
            'Invalid or overlapping IDs or classes')
    target[ids] = values


def accuracy_row(model, group, labels, pm, terminal, accepted, references):
    n = len(labels)
    correct = int((labels == pm).sum())
    row = dict(model=model, group=group, n=n, pm_correct=correct, pm_pct=100*correct/n,
               terminal_correct=int((labels == terminal).sum()), terminal_pct=100*np.mean(labels == terminal),
               accepted=int(accepted.sum()), accepted_pct=100*np.mean(accepted))
    for key, values in references.items():
        complete = np.all(values >= 0)
        count = int((labels == values).sum()) if complete else None
        row.update({key+'_correct': count, key+'_pct': 100*count/n if complete else None,
                    'pm_minus_'+key+'_pp': 100*(correct-count)/n if complete else None})
    return row


def csv_rows(data: bytes):
    return list(csv.DictReader(io.StringIO(data.decode())))


def historical_heads(archive, labels, label_sha, role, arrays):
    """Both nested archive trees are transitively SHA-pinned to accepted readbacks."""
    outer = json.loads(archive.read('config.json'))
    manifest = outer['manifest']
    timing = []
    names = [n for n in archive.namelist() if n.startswith('original_archives/') and n.endswith('.zip')]
    require(len(names) == len(CORRUPTIONS), 'Expected 15 historical corruption archives')
    seen = set()
    for name in names:
        with zipfile.ZipFile(io.BytesIO(archive.read(name))) as case, tempfile.TemporaryDirectory() as tmp:
            # Parent SHA is pinned; extraction here is only for the existing per-unit reader.
            case.extractall(tmp)
            predictions = {}
            verified = read_shard(Path(tmp), labels, label_sha, predictions=predictions,
                                  expected_campaign=outer['campaign'])
            require(verified['spec'] == manifest, 'Historical shard manifest mismatch')
            for row in csv_rows(case.read('per_corruption.csv')):
                if row['method'] in ('cloudfixer_plain', 'proximal_n8_pm'):
                    timing.append({k:row[k] for k in ('method','corruption','backbone',
                                  'runtime_seconds','reconstruction_seconds_shared','peak_gpu_memory_mb')})
        for (corr, model), samples in predictions.items():
            require((corr, model) not in seen and len(samples) == N, 'Historical scope mismatch')
            seen.add((corr, model))
            c, h = CORRUPTIONS.index(corr), MODELS.index(model)
            dense = np.array([samples[i] for i in range(N)])
            require(np.array_equal(dense[:, 0], labels), 'Historical label mismatch')
            if role == 'cf':
                arrays['cf'][c, :, h] = dense[:, 1]
            else:
                arrays['baseline_pm'][c, :, h] = dense[:, 1]
                arrays['baseline_terminal'][c, :, h] = dense[:, 2]
    require(len(seen) == 45, 'Missing historical corruption/head')
    return manifest, timing


def historical_pilot(path, lineage, labels, repeats):
    require(file_sha(path) == DEV_ZIP, 'Point-MAE development archive SHA mismatch')
    seconds = np.zeros((3, len(CORRUPTIONS)))
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        cfg, = [n for n in members if n.endswith('/config.json')]
        meta = json.loads(archive.read(cfg))
        manifest = meta['manifest']
        require(manifest['dataset_files'] == lineage['dataset_files'] and manifest['author_commit'] == lineage['author_commit']
                and manifest['diffusion_sha256'] == lineage['diffusion_sha256'], 'Point-MAE development data/prior mismatch')
        lineage['checkpoint_sha256']['pointmae'] = manifest['classifier_sha256']
        dev_ids = manifest['registration']['development_ids']
        require(len(set(dev_ids)) == len(dev_ids) == 128, 'Invalid development IDs')
        for name in members:
            if '/pending_unit_' not in name or not name.endswith('.npz'):
                continue
            with npz(archive.read(name)) as d:
                unit = json.loads(str(d['unit']))
                if unit['config']['method'] != 'cloudfixer':
                    continue
                rep, corr, ids = unit['config']['operational_repeat'], unit['corruption'], d['ids']
                require(rep in (0, 1, 2) and unit['phase'] == 'development' and unit['seed'] == 0
                        and unit['campaign'] == meta['campaign'] and np.array_equal(ids, unit['ids'])
                        and np.array_equal(labels[ids], d['labels']), 'Invalid historical Point-MAE pilot unit')
                state = str(d['model_hashes'][1])
                require(lineage['frozen_states'].setdefault('pointmae', state) == state, 'Point-MAE state changed')
                c = CORRUPTIONS.index(corr)
                logits = checked_logits(d['logits'], (len(ids), 1, 40))
                record_predictions(repeats[rep, c], ids, logits[:, 0].argmax(-1))
                seconds[rep, c] += float(d['seconds'])
        require(np.all(np.sum(repeats >= 0, axis=-1) == 128) and np.all(repeats[:, :, dev_ids] >= 0),
                'Historical pilot coverage differs across repeats')
    return seconds.tolist()


def recovered_cases(path, lineage, labels, target):
    ledger_path = path/'local_checkpoint_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    def saved_hash(archive):
        value = file_sha(archive)
        if ledger:
            require(ledger.get(archive.stem) == {'sha256':value, 'bytes':archive.stat().st_size},
                    'Historical recovery ledger mismatch: '+archive.name)
        return value

    with zipfile.ZipFile(path/'header.zip') as header:
        cfg = json.loads(header.read('config.json'))
        require(cfg['method'] == 'cloudfixer' and cfg['seed'] == 0 and cfg['severity'] == 5
                and cfg['args']['vote'] == 1 and cfg['args']['rotate'] and cfg['args']['classifier'] == 'pointMAE'
                and cfg['classifier_sha256'] == lineage['checkpoint_sha256']['pointmae']
                and cfg['diffusion_sha256'] == lineage['diffusion_sha256'] and cfg['dataset_sha256'] == lineage['dataset_files']
                and cfg['model_hashes'][0] == lineage['frozen_states']['pointmae']
                and cfg['author_commit'] == lineage['author_commit']
                and digest(header.read('executed_runner.py')) == cfg['runner_sha256'], 'Point-MAE recovery header mismatch')
    report = dict(run=cfg['run_id'], header_sha256=saved_hash(path/'header.zip'), runner_sha256=cfg['runner_sha256'],
                  complete={}, partial={}, batch_size=cfg['batch_size'])
    for c, corr in enumerate(CORRUPTIONS):
        chunks = sorted(path.glob(corr+'_batch_*.zip'))
        if not chunks:
            continue
        values = np.full(N, -1, dtype=np.int16)
        hashes = {}
        seconds = 0.
        for chunk in chunks:
            hashes[chunk.name] = saved_hash(chunk)
            with zipfile.ZipFile(chunk) as shard:
                for name in shard.namelist():
                    if not name.endswith('.npz'):
                        continue
                    require(name.startswith(corr+'_'), 'Mismatched Point-MAE chunk corruption')
                    with npz(shard.read(name)) as d:
                        ids = d['original_ids']
                        pred = checked_logits(d['logits'], (len(ids), 40)).argmax(-1)
                        require(np.array_equal(pred, d['prediction']) and np.array_equal(labels[ids], d['labels']),
                                'Point-MAE recovery labels or saved predictions differ')
                        record_predictions(values, ids, pred)
                        seconds += float(d['runtime_seconds'])
        complete = path/(corr+'_complete.zip')
        count = int((values >= 0).sum())
        if not complete.is_file() or count < N:
            report['partial'][corr] = dict(closed_ids=count, chunk_sha256=hashes)
            continue
        with zipfile.ZipFile(complete) as seal:
            row, = [r for r in csv_rows(seal.read('per_corruption.csv')) if r['corruption'] == corr]
            correct = int((values == labels).sum())
            require(row['status'] == 'complete' and row['run_id'] == cfg['run_id'] and int(row['n_examples']) == N
                    and int(row['n_correct']) == correct and np.isclose(float(row['runtime_seconds']), seconds, atol=1e-6, rtol=0),
                    'Point-MAE recovered counts or timing mismatch')
            state_file = corr+'_model_state.json'
            if state_file in seal.namelist():
                state = json.loads(seal.read(state_file))
                require(state['status'] == 'unchanged' and state['model_hashes'] == cfg['model_hashes'],
                        'Recovered Point-MAE changed model state')
            report['complete'][corr] = dict(correct=correct, n=N, seconds=float(row['runtime_seconds']),
                                            state_proof=state_file in seal.namelist(), chunk_sha256=hashes,
                                            complete_sha256=saved_hash(complete))
        # Prefer the first supplied complete archive for each corruption; keep alternate receipts.
        chosen = not np.all(target[c] >= 0)
        report['complete'][corr]['used_for_comparison'] = chosen
        if chosen:
            target[c] = values
        else:
            report['complete'][corr]['different_predictions_vs_selected'] = int((target[c] != values).sum())
    return report


def prepare_references(args):
    require(file_sha(args.baseline) == BASE_ZIP, 'Historical baseline final archive SHA mismatch')
    arrays = {k: np.full((15, N, 4), -1, dtype=np.int16) for k in ('cf', 'baseline_pm', 'baseline_terminal')}
    with zipfile.ZipFile(args.baseline) as base:
        plain_raw = base.read('comparison/final_20261122T230335Z.zip')
        require(digest(plain_raw) == PLAIN_ZIP, 'Historical plain CF final archive SHA mismatch')
        raw = base.read('comparison/label.npy')
        labels = np.load(io.BytesIO(raw), allow_pickle=False).reshape(-1)
        label_sha = digest(raw)
        require(labels.shape == (N,) and np.array_equal(np.unique(labels), np.arange(40)), 'Invalid canonical labels')
        with zipfile.ZipFile(io.BytesIO(plain_raw)) as plain:
            cf_spec, cf_timing = historical_heads(plain, labels, label_sha, 'cf', arrays)
        base_spec, base_timing = historical_heads(base, labels, label_sha, 'baseline', arrays)
    for k in ('head_configs','head_state_hashes','dataset_files','diffusion_sha256','author_commit','files'):
        require(cf_spec[k] == base_spec[k], 'Plain CF and baseline identities differ: '+k)
    lineage = {k: cf_spec[k] for k in ('head_configs','dataset_files','diffusion_sha256','author_commit')}
    lineage['checkpoint_sha256'] = {h: cf_spec['files'][cfg['checkpoint']]['sha256']
                                   for h,cfg in cf_spec['head_configs'].items()}
    lineage['frozen_states'] = dict(cf_spec['head_state_hashes'])
    repeats = np.full((3,15,N), -1, dtype=np.int16)
    pilot_seconds = historical_pilot(args.pilot, lineage, labels, repeats)
    cases = [recovered_cases(path, lineage, labels, arrays['cf'][:, :, 3]) for path in args.recovered]
    published = csv_rows((EVIDENCE/'cloudfixer_multibackbone_20261122/a100_per_corruption.csv').read_bytes())
    for row in (r for r in published if r['method'] == 'cloudfixer_plain'):
        c, h = CORRUPTIONS.index(row['corruption']), MODELS.index(row['backbone'])
        require(int(row['n_correct']) == (arrays['cf'][c,:,h] == labels).sum() and int(row['n_examples']) == N,
                'Raw plain CF disagrees with published all15 records')
    arrays.update(labels=labels, pilot=repeats)
    meta = dict(lineage=lineage, corruptions=list(CORRUPTIONS), heads=list(MODELS), sha256_baseline_bundle=BASE_ZIP,
                sha256_plain_bundle=PLAIN_ZIP, sha256_development_bundle=DEV_ZIP,
                recovery=cases, pilot_runtime_seconds=pilot_seconds,
                historical_timing=cf_timing+base_timing, protocol={'cf':cf_spec['scientific_protocol'],
                'baseline':base_spec['scientific_protocol']}, development_unique_ids_per_corruption=128)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out/'references.npz', **arrays)
    meta['predictions_sha256'] = file_sha(args.out/'references.npz')
    (args.out/'references.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps({'directory':str(args.out),'full_pointmae_cases':[k for x in cases for k in x['complete']],
                      'pilot_correct_by_repeat':((repeats == labels).sum(axis=(1,2))).tolist(),
                      'three_head_cf_correct':(arrays['cf'][:, :, :3] == labels[:,None]).sum(axis=(0,1)).tolist()}))


def groups():
    return [*((name, [i]) for i, name in enumerate(CORRUPTIONS)),
            ('other14', [i for i,c in enumerate(CORRUPTIONS) if c != 'rotation']),
            ('all15', list(range(len(CORRUPTIONS))))]


def read_candidate(path, expected_sha, reference, lineage):
    from scripts.run_cloudfixer_surface_four import verify_archive

    require(file_sha(path) == expected_sha, 'Candidate downloaded ZIP differs from independently recorded SHA')
    seal = verify_archive(path)
    require(seal['model_object_records'] == N*len(CORRUPTIONS)*len(MODELS), 'Missing candidate model/object records')
    pm = np.full((15, N, 4), -1, dtype=np.int16)
    terminal = np.full_like(pm, -1)
    accepted = np.zeros((15,N), dtype=bool)
    timing = {k:np.zeros((15,N)) for k in ('wall_ms', 'trajectory_ms', 'peak_mib', 'batch_size',
                                          'batch_start', *(m+'_head_ms' for m in MODELS))}
    with zipfile.ZipFile(path) as archive:
        campaign = json.loads(archive.read('campaign.json'))
        runtime = campaign['runtime']
        check_lineage(runtime, lineage)
        frozen = json.loads(archive.read('frozen.json'))
        require(frozen['states'] == runtime['frozen_states'], 'End-of-run frozen states differ from startup')
        require(np.array_equal(campaign['labels'], reference['labels']), 'Candidate label map differs from reference')
        attempts = {n:json.loads(archive.read(n)) for n in archive.namelist()
                    if n.startswith('attempts/') and n.endswith('.json')}
        for name in archive.namelist():
            if not name.startswith('blocks/') or not name.endswith('.npz'):
                continue
            with npz(archive.read(name)) as d:
                ids, corr = d['ids'], str(d['corruption'])
                c = CORRUPTIONS.index(corr)
                require(np.array_equal(reference['labels'][ids], d['labels']), 'Candidate/reference label mismatch')
                accepted[c,ids] = d['accepted']
                for field, values in timing.items():
                    values[c,ids] = d[field]
                for h, model in enumerate(MODELS):
                    pred, end = pm_and_terminal(d[model+'_logits'])
                    require(np.array_equal(pred, d[model+'_pm']) and np.array_equal(end, d[model+'_terminal']),
                            'CPU logit recount disagrees with saved '+model+' classes')
                    record_predictions(pm[c,:,h], ids, pred)
                    record_predictions(terminal[c,:,h], ids, end)
    require((pm >= 0).all() and (terminal >= 0).all(), 'Incomplete full candidate coverage')
    return pm, terminal, accepted, timing, campaign, attempts, seal


def check_lineage(runtime, lineage):
    require(runtime['dataset_files'] == lineage['dataset_files'] and runtime['author_commit'] == lineage['author_commit'],
            'Candidate and historical dataset or author revision differ')
    checkpoints, states = runtime['checkpoint_sha256'], runtime['frozen_states']
    require(all(checkpoints.get(h) == lineage['checkpoint_sha256'][h] for h in MODELS),
            'Candidate and historical classifier checkpoint digests differ')
    require(checkpoints['prior'] == lineage['diffusion_sha256'], 'Candidate prior checkpoint differs')
    require(all(states.get(h) == lineage['frozen_states'][h] for h in MODELS),
            'Candidate and historical frozen classifier states differ')
    require(set(runtime['heads']) == set(MODELS) and all(runtime['heads'].values()), 'Missing head coordinate/FPS contracts')
    class_names = json.loads((ROOT/'cfgs/cloudfixer_point2vec_assets.json').read_text())['class_order']
    require(runtime['class_order'] == class_names, 'Candidate class indices differ from canonical ModelNet40 order')


def timing_rows(timing, historical, sizes, recovery):
    records = []
    mae_seconds = {}
    for run in recovery:
        for corr, record in run['complete'].items():
            mae_seconds.setdefault(corr, record['seconds'])
    for group, ix in groups():
        n = len(ix)*N
        selected = {str(sizes[CORRUPTIONS[c]]) for c in ix}
        wall = timing['wall_ms'][ix]
        trajectory = timing['trajectory_ms'][ix]
        start_mask = timing['batch_start'][ix] == np.arange(N)
        batches = timing['batch_size'][ix][start_mask]
        pipeline_seconds = float(wall.sum()/1000)
        require(pipeline_seconds > 0 and (batches > 0).all() and int(batches.sum()) == n, 'Invalid active batch timing')
        batch_p50, batch_p95 = np.percentile(wall[start_mask]*batches, [50,95])
        for model in MODELS:
            head = timing[model+'_head_ms'][ix]
            item = dict(model=model, group=group, n=n, selected_batch_sizes=','.join(sorted(selected)),
                        timed_batches=int(start_mask.sum()), pipeline_all_heads_s=pipeline_seconds,
                        trajectory_shared_s=float(trajectory.sum()/1000), head_three_readouts_s=float(head.sum()/1000),
                        pipeline_objects_per_s=n/pipeline_seconds,
                        pipeline_amortized_p50_ms=float(np.median(wall)), pipeline_amortized_p95_ms=float(np.percentile(wall,95)),
                        pipeline_batch_p50_ms=float(batch_p50), pipeline_batch_p95_ms=float(batch_p95),
                        peak_allocated_mib=float(timing['peak_mib'][ix].max()))
            for key in ('cloudfixer_plain','proximal_n8_pm'):
                matches = [r for r in historical if r['method'] == key and r['backbone'] == model
                           and CORRUPTIONS.index(r['corruption']) in ix]
                item[key+'_shared_plus_head_s'] = sum(float(r['runtime_seconds']) for r in matches) if len(matches) == len(ix) else None
            available = model == 'pointmae' and all(CORRUPTIONS[c] in mae_seconds for c in ix)
            item['pointmae_plain_b32_pipeline_s'] = sum(mae_seconds[CORRUPTIONS[c]] for c in ix) if available else None
            records.append(item)
    return records


def pilot_rows(pm, terminal, accepted, labels, pilot):
    rows = []
    for group, ix in groups():
        subset = pilot[0,ix] >= 0
        y = np.broadcast_to(labels, subset.shape)[subset]
        pred, end = pm[ix,:,3][subset], terminal[ix,:,3][subset]
        counts = [(pilot[j,ix][subset] == y).sum() for j in range(3)]
        correct = int((y == pred).sum())
        row = dict(group=group, unique_id_corruption_pairs=len(y), pm_correct=correct,
                   terminal_correct=int((y == end).sum()), accepted=int(accepted[ix][subset].sum()),
                   pm_pct=100*correct/len(y), repeat0_correct=int(counts[0]), repeat1_correct=int(counts[1]),
                   repeat2_correct=int(counts[2]), cf_repeat_mean_pct=100*np.mean(counts)/len(y),
                   pm_minus_cf_repeat_mean_pp=100*(correct-np.mean(counts))/len(y))
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open('x', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def format_score(row, key):
    count = row[key+'_correct']
    return f"{count}/{row['n']} ({row[key+'_pct']:.2f}%)" if count is not None else 'Eksik referans'


def report_text(rows, timing, pilot, wall, runtime):
    lines = ['# ModelNet40-C • Yüzey onaylı PCA, severity 5, seed 0', '',
             'Her model 15 bozulmada aynı 2.468 özgün nesneyi değerlendirir. '
             'PM, t20/t10/gerçek t0 sınıf olasılıklarının eşit ortalamasıdır. '
             'Tüm sayımlar indirilen bloklar içindeki logitlerden üretildi. '
             'Tarihsel CF ve SCD ile karşılaştırmalar özgün ID ve etiket üzerinden eşlidir. '
             'Batch ve rastgele sayı akışı protokolleri farklıdır.', '']
    def delta(value):
        return '—' if value is None else f'{value:+.2f}'
    for model in MODELS:
        group = [r for r in rows if r['model'] == model]
        lines.extend([f'## {model}', '',
            '| Bozulma | Aday PM | Aday t0 | Plain CF | PM−CF (yp) | SCD PM | PM−SCD PM (yp) | SCD t0 | Kabul |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for row in [r for r in group if r['group'] != 'other14']:
            title = 'Overall accuracy' if row['group'] == 'all15' else row['group']
            cells = [title,format_score(row,'pm'),format_score(row,'terminal'),format_score(row,'cf'),
                     delta(row['pm_minus_cf_pp']),format_score(row,'baseline_pm'),
                     delta(row['pm_minus_baseline_pm_pp']), format_score(row,'baseline_terminal'),
                     f"{row['accepted']}/{row['n']} ({row['accepted_pct']:.2f}%)"]
            lines.append('| '+' | '.join(cells)+' |')
        other, = [r for r in group if r['group'] == 'other14']
        lines.extend(['',f"Rotation dışındaki 14 bozulmada PM {format_score(other,'pm')}, "
                      f"plain CF {format_score(other,'cf')}, SCD PM {format_score(other,'baseline_pm')}."])
        for label, key in (('Plain CF','cf'),('SCD PM','baseline_pm')):
            available = [r for r in group if r['group'] in CORRUPTIONS and r[key+'_correct'] is not None]
            if available:
                losers = [r['group'] for r in available if r['pm_minus_'+key+'_pp'] < 0]
                lines.append(f"{label} ile eşlenen {len(available)} bozulmada aday PM gerilemesi "
                             + (', '.join(losers) if losers else 'yok') + '.')
        lines.append('')
    lines.extend(['## Point-MAE • 128 özgün nesne üzerinde eşli karşılaştırma','',
        'Üç CF sütunu aynı 128 nesneyi tekrar kullanır. Her bozulmada 128 farklı karar vardır. '
        'All15 için 1.920 nesne–bozulma çifti ve 5.760 CF operasyonel tahmini vardır.', '',
        '| Bozulma | Eşli karar | Aday PM | Aday t0 | CF tekrar 0 | CF tekrar 1 | CF tekrar 2 | CF ort. % | PM−CF ort. (yp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in pilot:
        n = r['unique_id_corruption_pairs']
        cells = [r['group'],str(n),*(f'{r[k]}/{n}' for k in ('pm_correct','terminal_correct','repeat0_correct',
                 'repeat1_correct','repeat2_correct')),f"{r['cf_repeat_mean_pct']:.2f}",delta(r['pm_minus_cf_repeat_mean_pp'])]
        lines.append('| '+' | '.join(cells)+' |')
    full_time = [r for r in timing if r['group'] == 'all15']
    t = full_time[0]
    lines.extend(['','## A100 ölçüm süreleri','',
        f"Seçilen batch boyutları {t['selected_batch_sizes']}. Ortak geometri {t['trajectory_shared_s']:.2f} s. "
        f"Dört head ile ölçülen çıkarım {t['pipeline_all_heads_s']:.2f} s, {t['pipeline_objects_per_s']:.2f} nesne/s. "
        f"Örnek başına amorti p50/p95 {t['pipeline_amortized_p50_ms']:.2f}/{t['pipeline_amortized_p95_ms']:.2f} ms, "
        f"gerçek batch p50/p95 {t['pipeline_batch_p50_ms']:.2f}/{t['pipeline_batch_p95_ms']:.2f} ms. "
        f"Tepe CUDA allocation {t['peak_allocated_mib']:.1f} MiB.",
        f"Donanım kaydı `{json.dumps(runtime['hardware'], ensure_ascii=False)}`.",
        f"Ölçüm protokolü: {runtime['timing']}. Rastgelelik protokolü: {runtime['rng']}."])
    if wall['completed_attempt_elapsed_s']:
        lines.append(f"Tamamlanan komutların duvar süresi toplamı {wall['completed_attempt_elapsed_s']:.2f} s.")
    lines.extend(['', '| Model | Aday head, üç okuma (s) | Eski CF B64, geometri + head (s) | Eski SCD PM B96, geometri + head (s) |',
                  '|---|---:|---:|---:|'])
    for r in full_time:
        values = [r[k] for k in ('head_three_readouts_s','cloudfixer_plain_shared_plus_head_s','proximal_n8_pm_shared_plus_head_s')]
        lines.append('| '+r['model']+' | '+' | '.join('—' if v is None else f'{v:.2f}' for v in values)+' |')
    for r in timing:
        if r['group'] in CORRUPTIONS and r['pointmae_plain_b32_pipeline_s'] is not None:
            lines.append(f"\nPoint-MAE {r['group']} için tarihsel B32 plain CF "
                         f"{r['pointmae_plain_b32_pipeline_s']:.2f} s. "
                         f"Adayda ortak geometri {r['trajectory_shared_s']:.2f} s ve Point-MAE okumaları "
                         f"{r['head_three_readouts_s']:.2f} s sürdü.")
    lines.extend(['','Aday sürelerinde ortak geometri dört model için bir kez ölçüldü. Tarihsel CF ve SCD satırları üç head paylaşımıyla ölçülen ortak geometri ve ilgili head süresidir. Batch boyutu, okuma sayısı ve RNG düzeni koşullara özeldir. '
        'Tek seed üzerindeki farklar bu donmuş modeller ve bu 2.468 nesne için betimseldir.',''])
    return '\n'.join(lines)


def report(args):
    metadata = json.loads((args.references/'references.json').read_text())
    source = args.references/'references.npz'
    require(file_sha(source) == metadata['predictions_sha256'], 'Cached historical prediction SHA mismatch')
    with np.load(source, allow_pickle=False) as ref:
        pm, terminal, accepted, timing, campaign, attempts, seal = read_candidate(
            args.archive, args.sha256, ref, metadata['lineage'])
        labels = ref['labels']
        refs = {k:ref[k] for k in ('cf','baseline_pm','baseline_terminal')}
        rows = []
        for h, model in enumerate(MODELS):
            for group, ix in groups():
                gold = np.broadcast_to(labels, (len(ix),N)).ravel()
                row = accuracy_row(model, group, gold, pm[ix,:,h].ravel(), terminal[ix,:,h].ravel(),
                                   accepted[ix].ravel(), {k:v[ix,:,h].ravel() for k,v in refs.items()})
                if group in CORRUPTIONS:
                    for k in ('n','pm_correct','terminal_correct','accepted'):
                        require(row[k] == seal['cells'][model][group][k], 'Independent logit count differs from runner')
                rows.append(row)
        pilot = pilot_rows(pm, terminal, accepted, labels, ref['pilot'])
    elapsed = sum(a['elapsed_seconds'] for a in attempts.values() if a.get('status') == 'complete')
    wall = dict(completed_attempt_elapsed_s=elapsed, attempts=attempts)
    timed = timing_rows(timing, metadata['historical_timing'], seal['selected_batch_sizes'], metadata['recovery'])
    args.out.mkdir(parents=True, exist_ok=False)
    write_csv(args.out/'per_corruption.csv', [r for r in rows if r['group'] in CORRUPTIONS])
    write_csv(args.out/'summary.csv', [r for r in rows if r['group'] not in CORRUPTIONS])
    write_csv(args.out/'runtime.csv', timed)
    write_csv(args.out/'pointmae_paired128.csv', pilot)
    (args.out/'report.md').write_text(report_text(rows, timed, pilot, wall, campaign['runtime']))
    record = dict(archive_sha256=args.sha256, archive_name=args.archive.name, provenance_sha256=seal['provenance_sha256'],
                  reference_json_sha256=file_sha(args.references/'references.json'), reference_npz_sha256=file_sha(source),
                  model_object_records=seal['model_object_records'], selected_batch_sizes=seal['selected_batch_sizes'],
                  protocol=campaign['protocol'], runtime=campaign['runtime'], wall=wall,
                  historical_protocol=metadata['protocol'], pointmae_recovery=metadata['recovery'])
    (args.out/'recount.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({'report':str(args.out/'report.md'),'model_object_records':seal['model_object_records']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('references', help='Recount the accepted historical raw bundles once')
    prep.add_argument('--baseline', required=True, type=Path, help='Accepted B96 final ZIP containing historical plain CF and label.npy')
    prep.add_argument('--pilot', required=True, type=Path, help='Point-MAE 128-object development ZIP')
    prep.add_argument('--recovered', required=True, type=Path, nargs='+', help='Point-MAE header + closed batch ZIP folders in descending priority for overlapping complete cases')
    prep.add_argument('--out', required=True, type=Path, help='New directory for ID-level reference predictions and lineage')
    compare = commands.add_parser('report', help='Verify a downloaded candidate ZIP, match IDs and write final tables')
    compare.add_argument('--archive', required=True, type=Path, help='Drive-readback ZIP with adjacent .zip.sha256')
    compare.add_argument('--sha256', required=True, help='Expected raw ZIP SHA-256 from the original publication receipt')
    compare.add_argument('--references', required=True, type=Path)
    compare.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    (prepare_references if args.action == 'references' else report)(args)


if __name__ == '__main__':
    main()
