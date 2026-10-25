import importlib.util,json
from pathlib import Path
root=Path('/content/studies/guidance-components')
path=root/'sensitivity_phases/sensitivity_campaign_controller.py'
spec=importlib.util.spec_from_file_location('sensitivity_cpu_integration',path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sha='4f633535772d84a42e40e99b77becc60e844a3ecfbd0473678867be6109289c1'
p,registry,selections=c.plan_verified(root/'sensitivity_campaign_plan.json',sha)
receipt=c.cpu_preflight_verified(p)
assert receipt['state']['strict_loads']==16
assert not (root/'sensitivity_gpu_approval.json').exists()
projected=c.campaign_budget_verified(p,'calibrate',0)
assert len(registry)==19 and len(selections)==6 and projected<=19800
print('AGENT_B_SENSITIVITY_PLAN_CPU_INTEGRATION_COMPLETE',flush=True)
