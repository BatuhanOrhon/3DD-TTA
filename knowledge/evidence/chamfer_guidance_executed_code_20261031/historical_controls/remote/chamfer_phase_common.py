"""Bind reviewed house helpers to this task; no work occurs during import."""
import json
from pathlib import Path
import shutil
import sys

PHASES = Path(__file__).resolve().parent
sys.path.insert(0, str(PHASES))
import deploy_assets_template as house

ROOT = Path('/content/studies/chamfer-guidance-variants')
DRIVE = Path('/content/drive/MyDrive/thesis/result/chamfer_guidance_variants_20261030')
SESSION = 'tta-chamfer-guidance-variants-a100-20261030-r1'
SETUP_REPORT = Path('/content/cli_setup_verification.json')
NATIVE_MARKERS = Path('/content/3DD-TTA/.colab_setup/native')
NATIVE_PHASES = ('prepare', 'emd', 'chamfer', 'pointnet2', 'pvcnn-smoke')
NATIVE_CHECKS = ['CUDA tensor sum', 'FPS', 'KNN', 'Chamfer identical input', 'EMD finite output']


def configure(receipt=None):
    support = json.loads((PHASES / 'support_identity.json').read_text())
    for name, record in support['support_files'].items():
        house.verify(PHASES / name, record['copied'])
    receipt = receipt or json.loads((PHASES / 'preparation_identity.json').read_text())
    house.require(receipt['status'] == 'sealed_CPU_only', 'Final source seal required')
    house.ROOT, house.DRIVE, house.SESSION = ROOT, DRIVE, SESSION
    house.PREPARED, house.ASSETS = ROOT / 'prepared_v1', ROOT / 'prepared_v1/assets'
    house.MANIFEST_SHA, house.SOURCE_SHA = receipt['manifest']['sha256'], receipt['source_snapshot']['sha256']
    house.DELTA, house.PAYLOAD = PHASES / 'chamfer_guidance_variants_v1_delta.zip', receipt['payload']
    def preserve(output):
        for path in sorted(PHASES.iterdir()):
            if path.is_file() and path.suffix in ('.py', '.json'):
                shutil.copyfile(path, output / path.name)
    house.preserve_phase_sources = preserve
    return house


def setup_gate():
    """Accept only the observed ten-cell run with its single approved deferral."""
    acceptance_path = PHASES / 'setup_acceptance.json'
    acceptance = json.loads(acceptance_path.read_text())
    wrappers = json.loads((PHASES / 'support_identity.json').read_text())['setup_wrappers']
    house.require(acceptance['session'] == SESSION and acceptance['status'] == 'passed_with_known_cell2_deferral', 'Owned setup acceptance required')
    house.require(set(acceptance['cells']) == {str(i) for i in range(1, 11)}, 'Exact ten-cell evidence required')
    house.require(acceptance['deferrals'] == [{'cell': 2, 'package': 'fastrlock', 'scope': 'pip_failure_only'}], 'Only known cell2 fastrlock failure is deferred')
    for i in range(1, 11):
        cell = acceptance['cells'][str(i)]
        house.require(cell['wrapper'] == wrappers['cell_%02d.py' % i], 'Setup wrapper identity mismatch')
        if i == 1:
            house.require(cell['status'] == 'passed' and cell['kernel_reconnected'] is True, 'Cell1 same-VM reconnection proof required')
        elif i == 2:
            house.require(cell['status'] == 'deferred_fastrlock_only'
                          and cell['marker'] == 'CHAMFER_VARIANTS_CELL02_FASTRLOCK_DEFERRED_ENV_VERIFIED'
                          and cell['deferral_verifier'] == wrappers['verify_cell02_deferred.py'], 'Cell2 deferral evidence mismatch')
        else:
            house.require(cell['status'] == 'passed' and cell['marker'] == 'CHAMFER_VARIANTS_SETUP_CELL_%02d_COMPLETE' % i,
                          'Exact setup completion marker missing')
    house.require(acceptance['verification']['path'] == str(SETUP_REPORT), 'Setup report path mismatch')
    house.verify(SETUP_REPORT, acceptance['verification'])
    text = SETUP_REPORT.read_text().rstrip()
    # The frozen cell10 verifier appends one literal backslash-n after its JSON.
    report = json.loads(text[:-2] if text.endswith('\\n') else text)
    house.require(report['status'] == 'passed' and report['python'].startswith('3.8')
                  and report['torch'] == '2.1.2+cu121' and report['torch_cuda'] == '12.1'
                  and 'A100' in report['gpu'] and report['native_checks'] == NATIVE_CHECKS
                  and report['modelnet_corruption_arrays'] >= 15, 'Cell10 native/data/checkpoint verification did not pass')
    for phase in NATIVE_PHASES:
        house.require((NATIVE_MARKERS / (phase + '.complete')).is_file(), 'Fresh native phase incomplete: ' + phase)
    return dict(acceptance=house.identity(acceptance_path), verification=house.identity(SETUP_REPORT),
                native_phase_markers=[str(NATIVE_MARKERS / (phase + '.complete')) for phase in NATIVE_PHASES])


def self_check():
    """Dry binding only: no /content writes, checkpoints, native imports or CUDA."""
    h = configure(dict(status='sealed_CPU_only', manifest={'sha256': 'a' * 64},
                       source_snapshot={'sha256': 'b' * 64}, payload={'bytes': 1, 'sha256': 'c' * 64}))
    h.require(h.ROOT == ROOT and h.ASSETS == ROOT / 'prepared_v1/assets' and h.DRIVE == DRIVE, 'Task paths not rebound')
    import preflight_components_cpu as cpu
    h.require(cpu.ROOT == ROOT and cpu.ASSETS == h.ASSETS and cpu.SESSION == SESSION, 'CPU proof imported peer globals')
    h.require("'--component-arm', 'chamfer'" in (PHASES / 'preflight_components_cpu.py').read_text(), 'CPU proof must use Chamfer-only')
    h.self_check()
    cpu.self_check()
    print('CHAMFER_PHASE_WRAPPER_DRY_SELF_CHECK_COMPLETE')


if __name__ == '__main__':
    house.require(sys.argv[1:] == ['--self-check'], 'Use --self-check; deployment has its own entrypoint')
    self_check()
