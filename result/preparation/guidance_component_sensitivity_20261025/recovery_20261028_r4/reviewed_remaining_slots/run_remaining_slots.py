"""Conda wrapper; require an explicit reviewed recovery-plan hash."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--recovery-plan-sha256', required=True); args = parser.parse_args()
    here = Path(__file__).resolve().parent
    plan_path = here / 'reviewed_recovery_plan.json'
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.recovery_plan_sha256:
        raise ValueError('Reviewed recovery plan differs before Conda launch')
    plan = json.loads(raw)
    for name, expected in plan['sources'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe reviewed source name')
        data = (here / name).read_bytes()
        if dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest()) != expected:
            raise ValueError('Reviewed source differs before Conda launch')
    logs = here / 'wrapper_logs'; logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    command = ['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env', 'python', '-u',
               str(here / 'remaining_slots_controller.py'), '--recovery-plan-sha256', args.recovery_plan_sha256]
    with (logs / (stamp + '_remaining_slots.log')).open('x') as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1')) as process:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            if process.wait() != 0:
                raise RuntimeError('Remaining slots stopped; preserve evidence and ledger; no automatic retry')
    print('AGENT_B_REVIEWED_REMAINING_SLOTS_RECORDED', flush=True)


if __name__ == '__main__':
    main()
