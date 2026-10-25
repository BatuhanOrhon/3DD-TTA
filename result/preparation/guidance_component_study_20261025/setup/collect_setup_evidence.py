"""Preserve the completed ten-cell setup; no research inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

root = Path('/content/studies/guidance-components')
(root / 'phases').mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output = root / ('setup_evidence_' + stamp)
output.mkdir(exist_ok=False)
drive = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/setup') / output.name
assert Path('/content/drive/MyDrive').is_dir()
drive.mkdir(parents=True, exist_ok=False)


def identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            digest.update(data)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


report = Path('/content/cli_setup_verification.json')
verified = json.loads(report.read_text())
assert verified['status'] == 'passed' and verified['torch'] == '2.1.2+cu121'
assert verified['torch_cuda'] == '12.1' and verified['modelnet_corruption_arrays'] == 15
sources = Path('/content/agent_b_setup_local_sources.zip')
assert identity(sources) == dict(bytes=44493, sha256='95f6524fedb4f9f30f431a7a7aa350aeeb36f3a9c6dd96276d8e6d10d3e80c06')
paths = [report, sources, Path('/content/collect_agent_b_setup.py'),
         Path('/content/3DD-TTA/scripts/build_cuda_extensions_colab.sh'),
         Path('/content/3DD-TTA/scripts/verify_colab_environment.py')]
for phase in ('prepare', 'emd', 'chamfer', 'pointnet2', 'pvcnn-smoke'):
    paths.append(Path('/content/3DD-TTA/.colab_setup/native') / (phase + '.complete'))
for path in paths:
    assert path.is_file() and not path.is_symlink()
    shutil.copyfile(path, output / path.name)
    assert identity(path) == identity(output / path.name)
(output / 'pip_freeze.txt').write_text(subprocess.check_output(
    ['/usr/local/envs/3dd_tta_env/bin/python', '-m', 'pip', 'freeze'], text=True))
manifest = {p.name: identity(p) for p in sorted(output.iterdir()) if p.is_file()}
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
archive = output.with_suffix('.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
    for path in sorted(output.iterdir()):
        stream.write(path, path.name)
with zipfile.ZipFile(archive) as stream:
    assert stream.testzip() is None
target = drive / archive.name
shutil.copyfile(archive, target)
assert identity(archive) == identity(target)
receipt = dict(status='complete', session='tta-guidance-components-a100-20261025-r1',
               utc=stamp, research_inference=False, setup_cells=10,
               archive=dict(identity(archive), local_path=str(archive), drive_path=str(target)))
(root / 'setup_evidence_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
(drive / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('AGENT_B_SETUP_EVIDENCE_RECEIPT', json.dumps(receipt), flush=True)
print('AGENT_B_SETUP_EVIDENCE_PERSISTED', flush=True)
