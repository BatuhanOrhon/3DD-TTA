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
    _tta_result = _tta_ip.run_cell('import condacolab\ncondacolab.check()\n\n# Projeyi klonla ve dev branşına geç\n!git clone https://github.com/BatuhanOrhon/3DD-TTA.git\n%cd 3DD-TTA\n!git checkout dev\n\n# 1. Yazarın 3dd_tta_env (Python 3.8) ortamını env.yaml dosyasından yaratıyoruz:\n!conda env create -f env.yaml', store_history=False)
finally:
    _tta_ip.system = _tta_previous_system
if _tta_result.error_before_exec or _tta_result.error_in_exec:
    raise RuntimeError('Setup cell 2 Python/shell execution failed') from (_tta_result.error_before_exec or _tta_result.error_in_exec)
print('AGENT_B_SETUP_CELL_02_COMPLETE', flush=True)
