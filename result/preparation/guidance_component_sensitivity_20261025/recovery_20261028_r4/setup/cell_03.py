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
    _tta_result = _tta_ip.run_cell('from google.colab import drive\nimport shutil\nimport os\n\ndrive.mount("/content/drive")\n\nsource_path = "/content/drive/MyDrive/thesis/KNN_CUDA-0.2-py3-none-any.whl"\ndest_path = "/content/KNN_CUDA-0.2-py3-none-any.whl"\n\nif os.path.exists(source_path):\n    shutil.copy(source_path, dest_path)\n    print(f"✅ Dosya başarıyla kopyalandı: {dest_path}")\nelse:\n    print(f"❌ HATA: Kaynak dosya bulunamadı: {source_path}")', store_history=False)
finally:
    _tta_ip.system = _tta_previous_system
if _tta_result.error_before_exec or _tta_result.error_in_exec:
    raise RuntimeError('Setup cell 3 Python/shell execution failed') from (_tta_result.error_before_exec or _tta_result.error_in_exec)
print('AGENT_B_SETUP_CELL_03_COMPLETE', flush=True)
