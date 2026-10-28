from pathlib import Path
import os
os.chdir('/content/3DD-TTA' if Path('/content/3DD-TTA').is_dir() else '/content')
_tta_ip = get_ipython()
_tta_previous_system = _tta_ip.system
def _tta_checked_system(command):
    result = _tta_previous_system(command)
    code = _tta_ip.user_ns.get('_exit_code', 0)
    if code:
        raise RuntimeError('Setup shell command failed with code %s: %s' % (code, command))
    return result
_tta_ip.system = _tta_checked_system
try:
    _tta_result = _tta_ip.run_cell('from datetime import datetime, timezone\nimport subprocess\n\nprint(datetime.now(timezone.utc).isoformat(), "Cell4e/PVCNN+GPU smoke başlıyor", flush=True)\n_tta_run_checked(["bash", "/content/3DD-TTA/scripts/build_cuda_extensions_colab.sh", "pvcnn-smoke"])\nprint(datetime.now(timezone.utc).isoformat(), "Cell4e/PVCNN+GPU smoke tamamlandı", flush=True)\n', store_history=False)
finally:
    _tta_ip.system = _tta_previous_system
if _tta_result.error_before_exec or _tta_result.error_in_exec:
    raise RuntimeError('Setup cell 8 Python/shell execution failed') from (_tta_result.error_before_exec or _tta_result.error_in_exec)
assert Path('/content/3DD-TTA/.colab_setup/native/pvcnn-smoke.complete').is_file(), 'Native completion marker missing'
import hashlib, json, shutil
from datetime import datetime, timezone
_tta_setup_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
_tta_local_progress = Path('/content/studies/guidance-components/setup') / (_tta_setup_stamp + '_cell08')
_tta_local_progress.mkdir(parents=True, exist_ok=False)
_tta_drive_progress = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/setup') / _tta_local_progress.name
assert Path('/content/drive/MyDrive').is_dir()
_tta_drive_progress.mkdir(parents=True, exist_ok=False)
_tta_setup_receipt = dict(status='verified', cell=8, utc=_tta_setup_stamp, session='tta-guidance-components-a100-20261028-r4', notebook_source_sha256='7fd5b4dd730dbe83e1d5f6e1f6c0f16c535f9230ba99306afcf04cf0dcc0b34c', research_inference=False, markers=[q.name for q in Path('/content/3DD-TTA/.colab_setup/native').glob('*.complete')])
(_tta_local_progress / 'receipt.json').write_text(json.dumps(_tta_setup_receipt, indent=2)+'\n')
shutil.copyfile(_tta_local_progress / 'receipt.json', _tta_drive_progress / 'receipt.json')
assert hashlib.sha256((_tta_local_progress / 'receipt.json').read_bytes()).hexdigest() == hashlib.sha256((_tta_drive_progress / 'receipt.json').read_bytes()).hexdigest()
print('AGENT_B_SETUP_DRIVE_PROGRESS_VERIFIED', _tta_setup_stamp, flush=True)
print('AGENT_B_SETUP_CELL_08_COMPLETE', flush=True)
