"""Continue a frozen study after a timed-out, unaccepted confirmation attempt."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def identity(path):
    path=Path(path)
    return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def install_recovery(controller,record,receipt_path,source_path):
    if record['per_job_timeout_seconds']!=1800:
        raise ValueError('Confirmation recovery is limited to1800seconds per job')
    original_jobs=controller.phase_jobs
    original_worker=controller.operations.run_worker
    original_evidence=controller.evidence
    def jobs(phase,root):
        values=original_jobs(phase,root)
        if phase=='C':
            if not values or values[0]!=record['failed_job']:
                raise ValueError('Frozen confirmation candidate changed')
            values=[dict(values[0],slot=record['retry_slot']),*values[1:]]
        return values
    def worker(command,cwd,console,timeout,slot):
        # The outer supervisor retains the original cumulative deadline.
        if timeout==900 and slot.startswith(('confirmation-','timing-')):
            timeout=record['per_job_timeout_seconds']
        return original_worker(command,cwd,console,timeout,slot)
    def evidence(root,values):
        result=original_evidence(root,values)
        if any(j['stage'] in ('confirmation','timing') for j in values):
            result.update({Path(p).name:identity(p) for p in (receipt_path,source_path)})
        return result
    controller.phase_jobs=jobs
    controller.operations.run_worker=worker
    controller.evidence=evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('controller','plan','approval','duration-amendment','recovery'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();record=json.loads(args.recovery.read_text())
    if identity(args.controller)!=record['controller_identity'] or identity(__file__)!=record['recovery_source_identity']:
        raise ValueError('Unregistered recovery/controller source')
    plan=json.loads(args.plan.read_text());root=Path(plan['root'])
    spec=importlib.util.spec_from_file_location('_frozen_duration_controller',args.controller)
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    if controller.study.canonical_sha(plan)!=record['plan_sha256']:
        raise ValueError('Recovery plan differs from frozen study')
    failed=root/'slots'/record['failed_job']['slot']
    if identity(failed/'failure.json')!=record['original_failure_identity'] or (failed/'receipt.json').exists():
        raise ValueError('Recovery requires an unchanged unaccepted failure')
    failure=json.loads((failed/'failure.json').read_text())
    if 'TimeoutExpired' not in failure['traceback'] or '900 seconds' not in failure['traceback']:
        raise ValueError('Not the registered900-second timeout')
    if record['retry_slot']!=record['failed_job']['slot']+'-retry1':
        raise ValueError('Unregistered retry slot')
    receipt=root/'confirmation_timeout_recovery.json';source=root/'confirmation_timeout_recovery.py'
    controller.freeze(root,receipt.name,record)
    if source.exists():
        if identity(source)!=identity(__file__):raise ValueError('Immutable recovery source changed')
    else:
        with source.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    for path in (receipt,source):controller.persist(path,controller.publication_root(plan)/path.name)
    install_recovery(controller,record,receipt,source)
    controller.run(plan,json.loads(args.approval.read_text()),'all',
                   duration_amendment=json.loads(args.duration_amendment.read_text()))


if __name__=='__main__':main()
