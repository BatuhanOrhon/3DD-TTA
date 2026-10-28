from pathlib import Path
import hashlib
r=Path('/content/studies/guidance-components')
assert hashlib.sha256((r/'sensitivity_campaign_plan.json').read_bytes()).hexdigest()=='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
assert hashlib.sha256((r/'sensitivity_confirmation_receipt.json').read_bytes()).hexdigest()=='5ab88805d14edbc9713275816aa11bef1d5e80784ea385b477199f7d6b779754'
assert hashlib.sha256((r/'sensitivity_campaign_r4_ledger.json').read_bytes()).hexdigest()=='60c8ba55d787c8e1e9ea4fa34f27eeb3e9aa8522300252c6c74863e1cad2736b'
p=r/'transport_helpers';assert not p.exists();p.mkdir()
print('AGENT_B_FAILURE_HELPER_DIRECTORY_READY',flush=True)
