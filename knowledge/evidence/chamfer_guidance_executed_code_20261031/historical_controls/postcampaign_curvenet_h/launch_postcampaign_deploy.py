from pathlib import Path
import hashlib,json,subprocess
refs={'/content/studies/scan-curvenet-h-postcampaign-20261030/remote/launch_scan_curvenet_h.py': 'ba9d611342e6fa9aea0b32f38ab1bd66ede080681c59b8928dedc99072bea14d', '/content/studies/scan-curvenet-h-postcampaign-20261030/remote/scan_curvenet_h_controls.py': 'fd766090fe343bb5eaa53dc76159cef59d83363f186c6e87155f063fc318ead2', '/content/studies/scan-curvenet-h-postcampaign-20261030/remote/phase_plan.json': '297badb31d81667e86467d84d0d2f35912b3e3546d234f7a75bdf5435b0f9d46'}
for path,expected in refs.items():
 assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, path
command=['/usr/local/envs/3dd_tta_env/bin/python', '-u', '/content/studies/scan-curvenet-h-postcampaign-20261030/remote/launch_scan_curvenet_h.py', '--plan', '/content/studies/scan-curvenet-h-postcampaign-20261030/remote/phase_plan.json', '--plan-sha256', '297badb31d81667e86467d84d0d2f35912b3e3546d234f7a75bdf5435b0f9d46', '--stage', 'deploy']
with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as child:
 for line in child.stdout:print(line,end="",flush=True)
 if child.wait():raise subprocess.CalledProcessError(child.returncode,command)
r=json.loads(Path('/content/studies/scan-curvenet-h-postcampaign-20261030/deploy_receipt.json').read_text())
assert r["state"]["status"]=="complete" and r["state"]["plan_sha256"]=='297badb31d81667e86467d84d0d2f35912b3e3546d234f7a75bdf5435b0f9d46'
assert r["state"]["cpu_preflight"]["cuda_initialized"] is False and r["state"]["cpu_preflight"]["strict_teacher_load"] is True
print('SCAN_CURVENET_POSTCAMPAIGN_DEPLOY_COMPLETE',flush=True)
