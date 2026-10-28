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
    _tta_result = _tta_ip.run_cell('from pathlib import Path\nimport shutil, urllib.request, zipfile\n\n# ModelNet40-C archive includes the clean data_original.npy and corruptions.\ndrive_dataset_dir = Path(\'/content/drive/MyDrive/thesis/dataset\')\nlocal_data_dir = Path(\'/content/3DD-TTA/data\')\nlocal_data_dir.mkdir(parents=True, exist_ok=True)\ncorrupted_zip_drive = drive_dataset_dir / \'modelnet40_c.zip\'\ncorrupted_zip_local = local_data_dir / \'modelnet40_c.zip\'\nif corrupted_zip_drive.is_file():\n    assert zipfile.is_zipfile(corrupted_zip_drive), \'Drive modelnet40_c.zip is not a valid ZIP\'\n    print(\'ModelNet40-C (clean + corruptions) Drive arşivinden kopyalanıyor...\', flush=True)\n    shutil.copyfile(corrupted_zip_drive, corrupted_zip_local)\nelse:\n    print(\'ModelNet40-C Zenodo kaynağından indiriliyor...\', flush=True)\n    partial = corrupted_zip_local.with_suffix(\'.zip.part\')\n    urllib.request.urlretrieve(\'https://zenodo.org/records/6017834/files/modelnet40_c.zip\', partial)\n    assert zipfile.is_zipfile(partial), \'Downloaded ModelNet40-C is not a valid ZIP\'\n    partial.replace(corrupted_zip_local)\n\nwith zipfile.ZipFile(corrupted_zip_local) as archive:\n    root = local_data_dir.resolve()\n    members = [item for item in archive.infolist() if item.filename.startswith("modelnet40_c/")]\n    assert members, "ModelNet40-C archive has no dataset folder"\n    for member in members:\n        destination = (root / member.filename).resolve()\n        assert root in destination.parents, \'Unsafe ModelNet40-C archive path\'\n        assert (member.external_attr >> 16) & 0o170000 != 0o120000, \'Archive symlinks are not allowed\'\n    assert archive.testzip() is None, \'ModelNet40-C ZIP CRC check failed\'\n    archive.extractall(local_data_dir, members=members)\nassert (local_data_dir / \'modelnet40_c/data_original.npy\').is_file(), \'Clean ModelNet40-C array is missing\'\nif not corrupted_zip_drive.exists():\n    drive_dataset_dir.mkdir(parents=True, exist_ok=True)\n    shutil.copyfile(corrupted_zip_local, corrupted_zip_drive)\nprint(\'ModelNet40-C temiz ve bozulmuş verileri hazır.\', flush=True)\n\n', store_history=False)
finally:
    _tta_ip.system = _tta_previous_system
if _tta_result.error_before_exec or _tta_result.error_in_exec:
    raise RuntimeError('Setup cell 9 Python/shell execution failed') from (_tta_result.error_before_exec or _tta_result.error_in_exec)
import hashlib, json, shutil
from datetime import datetime, timezone
_tta_setup_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
_tta_local_progress = Path('/content/studies/guidance-components/setup') / (_tta_setup_stamp + '_cell09')
_tta_local_progress.mkdir(parents=True, exist_ok=False)
_tta_drive_progress = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025/setup') / _tta_local_progress.name
assert Path('/content/drive/MyDrive').is_dir()
_tta_drive_progress.mkdir(parents=True, exist_ok=False)
_tta_setup_receipt = dict(status='verified', cell=9, utc=_tta_setup_stamp, session='tta-guidance-components-a100-20261028-r4', notebook_source_sha256='6b0977ebf431c1a7cc5a2b961778f65cc1ca79a145f71b9303a2aa0ec8a582b9', research_inference=False, markers=[q.name for q in Path('/content/3DD-TTA/.colab_setup/native').glob('*.complete')])
(_tta_local_progress / 'receipt.json').write_text(json.dumps(_tta_setup_receipt, indent=2)+'\n')
shutil.copyfile(_tta_local_progress / 'receipt.json', _tta_drive_progress / 'receipt.json')
assert hashlib.sha256((_tta_local_progress / 'receipt.json').read_bytes()).hexdigest() == hashlib.sha256((_tta_drive_progress / 'receipt.json').read_bytes()).hexdigest()
print('AGENT_B_SETUP_DRIVE_PROGRESS_VERIFIED', _tta_setup_stamp, flush=True)
print('AGENT_B_SETUP_CELL_09_COMPLETE', flush=True)
