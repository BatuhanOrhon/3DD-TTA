import importlib.util,json,shutil,subprocess,hashlib
from pathlib import Path
root=Path('/content/studies/guidance-components')
spec=importlib.util.spec_from_file_location('approved_campaign',root/'sensitivity_phases/sensitivity_campaign_controller.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sha='4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
p,registry,selections=c.plan_verified(root/'sensitivity_campaign_plan.json',sha)
assert c.cpu_preflight_verified(p)['state']['strict_loads']==16
assert c.campaign_budget_verified(p,'calibrate',0)<=19800
assert not (root/'sensitivity_campaign_ledger.json').exists(),'Unexpected existing campaign; inspect before work'
assert shutil.disk_usage(root).free>5*1024**3
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
assert 'A100' in hardware
approval={'status': 'approved', 'phase_plan_sha256': '4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1', 'session': 'tta-guidance-components-a100-20261025-r1', 'cap_seconds': 19800, 'user_reply': 'Tamam onaylıyorum', 'recorded_utc': '2026-10-26T05:52:12.022746+00:00', 'approved_scope': 'Registered whole sensitivity campaign: calibration, smoke, development, frozen confirmation; nominal4.1934h, margin5.3654h, hard5.5h; exact sealed plan'}
payload=json.dumps(approval,ensure_ascii=False,indent=2)+'\n'
path=root/'sensitivity_gpu_approval.json'
assert not path.exists(),'Approval already present; inspect provenance'
path.write_text(payload)
drive=c.DRIVE/'sensitivity_gpu_approval.json'
assert not drive.exists(),'Drive approval already present; inspect provenance'
drive.write_text(payload)
assert c.identity(path)==c.identity(drive)
c.approval_verified(sha)
print('AGENT_B_SENSITIVITY_APPROVAL_RECORDED',json.dumps(dict(approval=c.identity(path),hardware=hardware,free_disk_bytes=shutil.disk_usage(root).free)),flush=True)
print('AGENT_B_SENSITIVITY_APPROVAL_CPU_COMPLETE',flush=True)
