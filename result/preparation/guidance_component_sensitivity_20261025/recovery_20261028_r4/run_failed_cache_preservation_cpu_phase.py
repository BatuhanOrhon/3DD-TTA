"""Run failure/cache preservation in the pinned NumPy1/PyTorch CPU process."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path('/content/studies/guidance-components')
source=root/'transport_helpers/preserve_failed_cache_phase.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='77d073fad144faaf953ade9938ad07892c600fa5eef4e1c1ef2849cb1ba80f33'
command=['conda','run','--no-capture-output','-n','3dd_tta_env','python','-u',str(source)]
logs=root/'phase_command_logs';logs.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
with (logs/(stamp+'_failed_confirmation_cache_CPU.log')).open('x') as log:
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')) as process:
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        assert process.wait()==0,'Failure evidence preservation failed; retain all partial artifacts'
receipt=json.loads((root/'sensitivity_confirmation_failure_preservation_receipt.json').read_text())
assert receipt['status']=='verified_failure_preserved' and receipt['experimental_result_status']=='failed'
assert receipt['physical_cache_count']==180 and receipt['physical_cache_complete'] and not receipt['mismatch_detected']
assert receipt['integrity_verification']['cache_input_tensor_post_RNG_hashes_verified']==180
assert receipt['integrity_verification']['cuda_initialized'] is False
print('AGENT_B_CONFIRMATION_FAILURE_PRESERVATION_WRAPPER_COMPLETE',flush=True)
