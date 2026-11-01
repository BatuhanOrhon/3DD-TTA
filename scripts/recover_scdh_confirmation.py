"""Continue a frozen study after a timed-out, unaccepted confirmation attempt."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def identity(path):
    path=Path(path)
    return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


NATIVE_REBUILDS={
    'pointnet2_ops._ext':dict(bytes=266256,old='ce3d592c03ffb43213bb4d080af899cbec371ac46053671de9ffd0b0a8836bac',new='aee8c9b61f4a62de7044678c46c1ca3164bdb448da989816adf9049f9138f5f0'),
    'chamfer':dict(bytes=720640,old='5c2e52fc47daa29c2c34418a3759fbf1d200e804c777bc314a7fd4a7ef6acf92',new='5ab9c3ed87c39c0dcb68f0cc96c957338fb6cbf1ec576a9aa88782bd52b68149'),
}


def transport_calibration(original,original_identity,current_native):
    previous=original['binding']['native_extensions']
    if set(previous)!=set(current_native):raise ValueError('Changed native module set')
    changes={}
    for name,before in previous.items():
        after=current_native[name]
        if name in NATIVE_REBUILDS:
            expected=NATIVE_REBUILDS[name]
            if (before['sha256']!=expected['old'] or after['sha256']!=expected['new']
                    or before['bytes']!=expected['bytes'] or after['bytes']!=expected['bytes']
                    or before['path']!=after['path']):raise ValueError('Unregistered native rebuild: '+name)
            changes[name]=dict(old=copy.deepcopy(before),new=copy.deepcopy(after))
        elif before!=after:raise ValueError('Unregistered native change: '+name)
    result=copy.deepcopy(original)
    result['binding']['native_extensions']=copy.deepcopy(current_native)
    result['runtime_transport']=dict(original_calibration_identity=original_identity,
        original_binding=copy.deepcopy(original['binding']),native_changes=changes,
        coefficients_recomputed=False,native_bitwise_equality_claimed=False,
        binding_role='current deployment compatibility; calibration observations retain original_binding',
        limitation='Frozen numeric coefficients carried to a new native build; matched confirmation arms share that build')
    return result


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
    transport=record.get('runtime_transport')
    if transport:
        root=Path(receipt_path).parent
        frozen_plan=json.loads((root/'plan.json').read_text())
        original_command=controller.command
        original_execute=controller.execute_job
        original_publication=controller.publication_root
        original_scoped_evidence=controller.evidence
        def command(plan,job):
            values=original_command(plan,job)
            if job['stage'] in ('confirmation','timing') and job['arm']=='spectral':
                values[values.index('--calibration-report')+1]=str(root/transport['calibration_file'])
                values[values.index('--calibration-report-sha256')+1]=transport['calibration_identity']['sha256']
            return values
        def execute(plan,job,deadline,manifest):
            current=plan
            if job['stage'] in ('confirmation','timing'):
                current=dict(plan,cpu_preflight_receipt=str(root/transport['cpu_file']),
                    cpu_preflight_receipt_identity=transport['cpu_identity'])
            return original_execute(current,job,deadline,manifest)
        def transported_evidence(root,values):
            result=original_scoped_evidence(root,values)
            if any(j['stage'] in ('confirmation','timing') for j in values):
                result.update({name:identity(Path(root)/name) for name in (transport['calibration_file'],transport['cpu_file'])})
            return result
        controller.command=command
        controller.execute_job=execute
        controller.publication_root=lambda plan:original_publication(frozen_plan)
        controller.evidence=transported_evidence


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
    transport=record.get('runtime_transport')
    if transport:
        original=controller.frozen_file(root,'frozen_calibration.json','calibration')[1]
        cpu=root/transport['cpu_file'];calibration=root/transport['calibration_file']
        if identity(cpu)!=transport['cpu_identity'] or identity(calibration)!=transport['calibration_identity']:
            raise ValueError('Runtime transport proof changed')
        expected=transport_calibration(original,identity(root/'frozen_calibration.json'),json.loads(cpu.read_text())['native_extensions'])
        if json.loads(calibration.read_text())!=expected:raise ValueError('Frozen calibration coefficients or transport binding changed')
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
