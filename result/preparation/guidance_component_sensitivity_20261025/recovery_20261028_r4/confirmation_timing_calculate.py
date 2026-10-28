"""Local archived timing evidence only; no prediction analysis or remote access."""
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path('/Users/batuhan/Desktop/projects/3DD-TTA-worktrees/guidance-components')
BASE = REPO / 'result/preparation'
OLD = BASE / 'guidance_component_study_20261025'
NEW = BASE / 'guidance_component_sensitivity_20261025'
RECOVERY = NEW / 'recovery_20261028_r4'


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(bytes=Path(path).stat().st_size, sha256=digest.hexdigest())


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


source_paths = dict(old_development_receipt=OLD / 'development_phase_receipt.json',
                    old_confirmation_state=OLD / 'frozen_confirmation_extracted/state.json',
                    new_smoke_state=RECOVERY / 'local_smoke_analysis_v1/phase/state.json',
                    current_plan=RECOVERY / 'controls/bundle/campaign_plan.json',
                    prior_smoke_projection=RECOVERY / 'post_smoke_budget_projection.json')
sources = {name: dict(path=str(path), **identity(path)) for name, path in source_paths.items()}
plan = json.loads(source_paths['current_plan'].read_text())
prior = json.loads(source_paths['prior_smoke_projection'].read_text())
development = json.loads(source_paths['old_development_receipt'].read_text())['state']
confirmation = json.loads(source_paths['old_confirmation_state'].read_text())
smoke = json.loads(source_paths['new_smoke_state'].read_text())
assert sources['current_plan']['sha256'] == '1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
assert confirmation['status'] == smoke['status'] == development['status'] == 'complete'
ratio = prior['conservative_max_ratio']
assert ratio == 1.025455024723712

# Reproduce the sealed numerical estimate exactly. The source preparation script
# seals assets; it does not contain this timing arithmetic.
dev_rows = []
for job in development['jobs']:
    if job['arm'] not in ('combined', 'chamfer'):
        continue
    shared = sum(row['shared_reconstruction_seconds'] for row in job['timings'].values())
    dev_rows.append(dict(arm=job['arm'], dataset=job['dataset'], wall70=job['elapsed_seconds'],
                         shared70=shared, registered140=job['elapsed_seconds'] + shared))
registered = 3 * sum(row['registered140'] * (2 if row['arm'] == 'chamfer' else 1) for row in dev_rows) + 6 * 270
assert registered == plan['stage_expected_seconds']['confirmation'] == 7242.365194455003


def measured_rows(state, root):
    files = {path.name: path for path in root.rglob('*.zip')}
    rows, archives = [], []
    for job in state['jobs']:
        shared = sum(row['shared_reconstruction_seconds'] for row in job['timings'].values())
        head = 0.
        for raw in job['raw_archives']:
            path = files[Path(raw['local_path']).name]
            actual = identity(path)
            assert actual == {key: raw[key] for key in ('bytes', 'sha256')}
            archives.append(dict(path=str(path), **actual))
            with zipfile.ZipFile(path) as archive:
                head += sum(json.loads(archive.read(name))['head_seconds'] for name in archive.namelist()
                            if '/batches/' in name and name.endswith('/diagnostics.json'))
        fixed = job['elapsed_seconds'] - shared - head
        assert fixed >= 0 and head > 0
        rows.append(dict(arm=job.get('arm', job.get('condition_id')), dataset=job['dataset'], seed=job['seed'],
                         wall70=job['elapsed_seconds'], shared70=shared, terminal_heads70=head,
                         residual70=fixed, point140=2 * (shared + head) + fixed,
                         preparation_modes=sorted({row['preparation_mode'] for row in job['timings'].values()})))
    return rows, dict(count=len(archives), identity_rows_sha256=object_sha(archives))


old_rows, old_archives = measured_rows(confirmation, OLD / 'frozen_confirmation_extracted')
smoke_rows, smoke_archives = measured_rows(smoke, RECOVERY / 'local_smoke_analysis_v1/phase')
assert len(old_rows) == 12 and old_archives['count'] == 48
assert len(smoke_rows) == 14 and smoke_archives['count'] == 56
maxima = []
for arm in ('combined', 'chamfer'):
    for dataset in ('modelnet40_c', 'scanobjectnn_c'):
        selected = [row for row in old_rows if row['arm'] == arm and row['dataset'] == dataset]
        assert len(selected) == 3
        maxima.append(dict(arm=arm, dataset=dataset, point140_max=max(row['point140'] for row in selected),
                           residual70_max=max(row['residual70'] for row in selected),
                           maximizing_seed=max(selected, key=lambda row: row['point140'])['seed']))
gsd = 3 * sum(row['point140_max'] * (2 if row['arm'] == 'chamfer' else 1) for row in maxima)
point = ratio * gsd + 6 * 300
floor = registered * ratio + 180
result = dict(status='provisional_archived_evidence_calculation', sources=sources,
              registered_formula='3*sum_dataset[(wall70_CS+shared70_CS)+2*(wall70_C+shared70_C)]+6*270',
              registered_development_rows=dev_rows, reproduced_registered_seconds=registered,
              measured_formula='E140=2*(shared70+terminal_heads70)+(wall70-shared70-terminal_heads70)',
              old_confirmation_per_seed_rows=old_rows, old_confirmation_raw_archives=old_archives,
              new_smoke_per_job_rows=smoke_rows, new_smoke_raw_archives=smoke_archives,
              maxima=maxima, gsd18job_unscaled_seconds=gsd, prior_smoke_max_ratio=ratio,
              gsd18job_scaled_seconds=ratio*gsd, original6job_provisional_seconds=6*300,
              maximum_conditions=4, maximum_jobs=24, three_seeds=True, original_ids_per_dataset=140,
              point_formula='ratio*3*sum_dataset[max_seed_E140_CS+2*max_seed_E140_C]+6*300',
              measured_point_seconds=point, measured_point_minutes=point/60,
              existing_registered_floor_seconds=floor, conservative_point_basis_seconds=max(point, floor),
              upper_formula='final_actual_campaign_spent_seconds+1.2*max(measured_point,registered_floor)+1200',
              maximum_final_spent_to_fit_seconds=19800-1200-1.2*max(point,floor),
              current_7176_5_spent_scenario_upper_seconds=7176.5+1.2*max(point,floor)+1200,
              finalization='Requires genuine final development artifacts, selected freeze, actual ledger and any higher measured slowdown ratio.',
              limitations='Old CS jobs include fresh first-condition cache creation; C jobs replay. Retain maximum4 conditions/24jobs even if freeze deduplicates. Remaining doubled-artifact packing/I/O is not isolated by70-ID data and remains in20%margin+1200reserve. Original300s per dataset/seed is provisional. No guarantee of runtime or cap fit; no new research results inspected.')
print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
