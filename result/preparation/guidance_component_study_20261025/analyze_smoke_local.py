"""Independently validate smoke and separate fixed startup from corruption cost."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

PREP = Path(__file__).resolve().parent
REPO = PREP.parents[2]
sys.path.insert(0, str(REPO))
from scripts.analyze_scan_trajectory import analyze_bundle


def identity(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


receipt = json.loads((PREP / 'smoke_phase_receipt.json').read_text())
initial = json.loads((PREP / 'smoke_validation_initial.json').read_text())
state = receipt['state']
assert state['status'] == 'complete' and len(state['jobs']) == 8
output = PREP / 'local_smoke_analysis'
output.mkdir(exist_ok=False)
raw_root = Path(initial['output'])
extracted = PREP / 'smoke_extracted'
reports, timings, inputs = [], [], []
for number, job in enumerate(state['jobs']):
    assert job['status'] == 'complete'
    for raw in job['raw_archives']:
        path = extracted / Path(raw['local_path']).relative_to(raw_root)
        assert identity(path) == {key: raw[key] for key in ('bytes', 'sha256')}
        report = analyze_bundle(path, allow_partial=True)
        assert report['selection_complete'] and report['status'] == 'partial'
        assert len(report['per_corruption']) == 6 and all(row['n'] == 70 for row in report['per_corruption'])
        destination = REPO / 'result' / job['dataset'] / 'gsd_inspired_latent_spectral_guidance' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copyfile(path, destination)
        assert identity(destination) == identity(path)
        inputs.append(dict(path=str(destination.relative_to(REPO)), **identity(path)))
        reports.append(report)
    text = (extracted / ('job_%02d.log' % number)).read_text()
    matches = re.findall(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) - Point_MAE - INFO - .*divide point cloud', text, re.M)
    assert len(matches) == 1
    # The final head's constructor runs once before any corruption; two seconds cover timestamp slack.
    startup_floor = (datetime.strptime(matches[0], '%Y-%m-%d %H:%M:%S,%f')
                     - datetime.strptime(job['utc'], '%Y%m%dT%H%M%S%fZ')).total_seconds() - 2.
    normal = job['timings']['gaussian']['shared_reconstruction_seconds']
    background = job['timings']['background']['shared_reconstruction_seconds']
    overhead = job['elapsed_seconds'] - normal - background
    assert 0 < startup_floor < overhead
    # Scale all remaining overhead, including fixed finalization, conservatively by 15/2.
    expected = 14 * normal + background + startup_floor + 7.5 * (overhead - startup_floor) + 10.
    timings.append(dict(dataset=job['dataset'], arm=job['arm'], smoke_job_seconds=job['elapsed_seconds'],
                        normal_reconstruction_seconds=normal, background_reconstruction_seconds=background,
                        fixed_startup_lower_bound_seconds=startup_floor, residual_smoke_overhead_seconds=overhead-startup_floor,
                        projected_development_seconds=expected, upper_with_20pct_margin_seconds=1.2 * expected))
assert len(inputs) == len(reports) == 32
assert len({(r['dataset'], r['component_arm'], r['backbone']) for r in reports}) == 32
assert sum(row['n'] for report in reports for row in report['per_corruption'] if row['policy'] == 'terminal') == 4480
development = sum(row['projected_development_seconds'] for row in timings)
projection = dict(status='CPU_derived_from_accepted_smoke', manifest_sha256=initial['manifest_sha256'],
                  raw_smoke_evidence=initial['evidence'], smoke_spent_seconds=initial['spent_seconds'],
                  original_linear_startup_projection_seconds=initial['projected_development_seconds'],
                  projected_development_seconds=development,
                  projected_whole_active_phase_seconds=initial['spent_seconds'] + development,
                  upper_whole_with_20pct_development_margin_seconds=initial['spent_seconds'] + 1.2 * development,
                  hard_active_phase_cap_seconds=3300, per_job=timings,
                  method='Fixed startup lower bound from the last frozen-head constructor timestamp; scale all residual overhead by15/2; add10s/case for extra input reads; retain separate20% uncertainty margin.',
                  limitations='Timing projection, not speedup or accuracy evidence. Only two corruption timing probes. Setup/billing/CPU analysis time is separate. Hard cap remains55min; an overrun stops with partial evidence.')
assert projection['projected_whole_active_phase_seconds'] <= 3300
assert projection['upper_whole_with_20pct_development_margin_seconds'] < 3600
(output / 'timing_projection.json').write_text(json.dumps(projection, indent=2, sort_keys=True) + '\n')
(output / 'validation.json').write_text(json.dumps(dict(status='accepted_smoke_only', raw_inputs=inputs,
    reports=reports, n_candidate_predictions=17920, accuracy_scope='smoke; not benchmark', pairing='fresh encodes; no cross-arm latent identity claim'), indent=2) + '\n')
print(json.dumps(dict(raw_archives=len(inputs), candidate_predictions=17920,
    expected_whole_minutes=projection['projected_whole_active_phase_seconds']/60,
    upper_whole_minutes=projection['upper_whole_with_20pct_development_margin_seconds']/60)))
