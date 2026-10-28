from pathlib import Path
import hashlib,runpy,sys
root=Path('/content/studies/guidance-components')
assert hashlib.sha256((root/'sensitivity_campaign_plan.json').read_bytes()).hexdigest()=='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
assert hashlib.sha256((root/'sensitivity_campaign_r4_ledger.json').read_bytes()).hexdigest()=='60c8ba55d787c8e1e9ea4fa34f27eeb3e9aa8522300252c6c74863e1cad2736b'
directory=root/'reviewed_remaining_slots'
if not directory.exists(): directory.mkdir()
wrapper=directory/'run_remaining_slots.py'
assert hashlib.sha256(wrapper.read_bytes()).hexdigest()=='ac5bb60d537befff8df029a3a7cce1ad1286d9d4e07e37d2c0059567b4309931'
previous_argv=sys.argv
try:
 sys.argv=[str(wrapper),'--recovery-plan-sha256','13ff25249aec97eafd6c1fb6db33cd5e3d8462b79331a7cfdfba0e4a06540f6c']
 runpy.run_path(str(wrapper),run_name='__main__')
finally: sys.argv=previous_argv
