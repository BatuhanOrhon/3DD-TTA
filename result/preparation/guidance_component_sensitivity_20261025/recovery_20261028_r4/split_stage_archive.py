"""Bounded-memory CPU transport of an immutable completed r4 phase archive."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil

STAGE = 'development'  # The confirmation adaptation changes only this literal.
ROOT = Path('/content/studies/guidance-components')
DRIVE = Path('/content/drive/MyDrive/thesis/result/guidance_component_study_20261025')
SESSION = 'tta-guidance-components-a100-20261028-r4'
PLAN_SHA = '1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
MANIFEST_SHA = '599b8dcb4f25d5297e26a69036aa124660adf38ff8ee51f106726f9379cd8eef'
SOURCE_SHA = '73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8'
CHUNK_BYTES = 256 * 1024 * 1024
BUFFER_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bytes_identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def identity(path):
    digest = hashlib.sha256(); size = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(BUFFER_BYTES), b''):
            digest.update(block);size += len(block)
    return dict(bytes=size,sha256=digest.hexdigest())


def remote_owned(value, root):
    path = PurePosixPath(value); root = PurePosixPath(root)
    require(isinstance(value,str) and '\\' not in value and path.is_absolute() and '..' not in path.parts
            and path.as_posix() == value and root in path.parents, 'Foreign/noncanonical remote path')
    return path


def validate_phase_receipt(receipt, stage):
    require(stage in ('development','confirmation'), 'Unregistered transport stage')
    state = receipt['state']; archive = receipt['archive']
    require(state['status'] == 'complete' and state['stage'] == stage and state['session'] == SESSION
            and state['phase_plan_sha256'] == PLAN_SHA and state['manifest_sha256'] == MANIFEST_SHA
            and state['source_snapshot_sha256'] == SOURCE_SHA, 'Incomplete/foreign original phase receipt')
    require(type(archive['bytes']) is int and archive['bytes'] > 0 and re.fullmatch('[0-9a-f]{64}',archive['sha256']) is not None, 'Invalid original archive identity')
    local = remote_owned(archive['local_path'],ROOT); drive = remote_owned(archive['drive_path'],DRIVE)
    require(local.parent == PurePosixPath(ROOT) and local.name.startswith('sensitivity_'+stage+'_') and local.suffix == '.zip'
            and drive == PurePosixPath(DRIVE)/'phase-evidence'/local.stem/('sensitivity_'+stage+'_evidence.zip'), 'Original archive escaped its owned stage namespace')
    return archive


def split_archive(source, output, expected, chunk_bytes=CHUNK_BYTES):
    require(type(chunk_bytes) is int and 1 <= chunk_bytes <= CHUNK_BYTES, 'Part size exceeds256MiB bound')
    aggregate = hashlib.sha256(); total = 0; rows = []
    with Path(source).open('rb') as stream:
        while total < expected['bytes']:
            size = min(chunk_bytes,expected['bytes']-total); name = 'part%05d.bin' % len(rows)
            path = Path(output)/name; digest = hashlib.sha256(); written = 0
            with path.open('xb') as target:
                while written < size:
                    block = stream.read(min(BUFFER_BYTES,size-written))
                    require(bool(block),'Original Drive archive ended before its receipt length')
                    target.write(block);digest.update(block);aggregate.update(block);written += len(block)
            row = dict(index=len(rows),name=name,offset=total,bytes=written,sha256=digest.hexdigest())
            require(identity(path) == {key:row[key] for key in ('bytes','sha256')}, 'Persisted transport part differs from streamed bytes')
            rows.append(row);total += written
            print('AGENT_B_CHUNK_TRANSPORT_PROGRESS',len(rows),total,expected['bytes'],flush=True)
        require(stream.read(1) == b'','Original Drive archive exceeds its receipt length')
    require(total == expected['bytes'] and aggregate.hexdigest() == expected['sha256'], 'Aggregate transport differs from original archive')
    require(len(rows) == (total+chunk_bytes-1)//chunk_bytes, 'Transport part count mismatch')
    return rows


def validate_manifest(manifest, receipt_raw, plan_raw, stage):
    require(bytes_identity(plan_raw)['sha256'] == PLAN_SHA, 'Supplied scientific plan changed')
    plan = json.loads(plan_raw)
    require(plan['session'] == SESSION and plan['manifest_sha256'] == MANIFEST_SHA and plan['source_snapshot_sha256'] == SOURCE_SHA, 'Plan source/session mismatch')
    archive = validate_phase_receipt(json.loads(receipt_raw),stage)
    require(manifest['status'] == 'ready' and manifest['schema_version'] == 1 and manifest['stage'] == stage
            and manifest['session'] == SESSION and manifest['phase_plan_sha256'] == PLAN_SHA
            and manifest['manifest_sha256'] == MANIFEST_SHA and manifest['source_snapshot_sha256'] == SOURCE_SHA
            and manifest['original_archive'] == archive and manifest['original_phase_receipt'] == bytes_identity(receipt_raw), 'Transport manifest does not bind the genuine original receipt')
    transport = remote_owned(manifest['transport_root'],ROOT)
    require(transport.parent == PurePosixPath(ROOT) and transport.name.startswith('sensitivity_'+stage+'_transport_'), 'Foreign transport directory')
    bound = manifest['chunk_bytes']; rows = manifest['parts']
    require(type(bound) is int and 1 <= bound <= CHUNK_BYTES and type(manifest['part_count']) is int
            and manifest['part_count'] == len(rows) == (archive['bytes']+bound-1)//bound, 'Invalid part bound/count')
    offset = 0
    for index,row in enumerate(rows):
        name = 'part%05d.bin' % index
        require(type(row['index']) is int and row['index'] == index and row['name'] == name
                and row['remote_path'] == str(transport/name) and remote_owned(row['remote_path'],ROOT).parent == transport
                and type(row['offset']) is int and row['offset'] == offset
                and type(row['bytes']) is int and row['bytes'] == min(bound,archive['bytes']-offset)
                and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None, 'Out-of-order, foreign or malformed transport part')
        offset += row['bytes']
    require(offset == archive['bytes'], 'Parts do not cover the complete original archive')
    return archive


def main():
    controller = ROOT/'sensitivity_phases/sensitivity_campaign_controller.py'
    require(identity(controller)['sha256'] == '0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc','Pinned r4 controller changed')
    spec = importlib.util.spec_from_file_location('archive_transport_control',controller)
    c = importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    plan_path = ROOT/'sensitivity_campaign_plan.json'
    c.plan_verified(plan_path,PLAN_SHA)
    ledger = c.load_ledger(PLAN_SHA)
    receipt_path = c.owned(ROOT/('sensitivity_'+STAGE+'_receipt.json'))
    validation_path = c.owned(ROOT/('sensitivity_'+STAGE+'_validation.json'))
    receipt_raw = receipt_path.read_bytes();receipt = json.loads(receipt_raw);validation = json.loads(validation_path.read_text())
    archive = validate_phase_receipt(receipt,STAGE)
    attempts = [row for row in ledger['attempts'] if row['stage'] == STAGE]
    require(ledger['active'] is None and len(attempts) == 1 and attempts[0]['status'] == 'complete'
            and validation['status'] == 'validated' and validation['stage'] == STAGE
            and validation['phase_plan_sha256'] == PLAN_SHA and validation['manifest_sha256'] == MANIFEST_SHA
            and validation['source_snapshot_sha256'] == SOURCE_SHA and validation['evidence'] == archive
            and attempts[0]['output'] == validation['output'] and archive['local_path'] == validation['output']+'.zip', 'Original stage is not complete/validated/inactive')
    source = c.owned(archive['drive_path'],c.DRIVE);local = c.owned(archive['local_path'])
    c.verify(local,archive)  # Stream the existing local ZIP; never load its bytes into RAM.
    require(source.stat().st_size == archive['bytes'],'Original Drive archive size changed')
    require(shutil.disk_usage(ROOT).free >= archive['bytes']+CHUNK_BYTES,'Insufficient VM disk for immutable transport parts')
    pointer = ROOT/('sensitivity_'+STAGE+'_chunk_transport_receipt.json')
    require(not pointer.exists(),'Completed transport pointer exists; inspect it instead of overwriting')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = c.owned(ROOT/('sensitivity_'+STAGE+'_transport_'+stamp));output.mkdir(exist_ok=False)
    try:
        records = split_archive(source,output,archive)
        rows = [dict(row,remote_path=str(output/row['name'])) for row in records]
        require(c.load_ledger(PLAN_SHA) == ledger,'Campaign ledger changed during inactive CPU transport')
        manifest = dict(status='ready',schema_version=1,stage=STAGE,session=SESSION,phase_plan_sha256=PLAN_SHA,
                        manifest_sha256=MANIFEST_SHA,source_snapshot_sha256=SOURCE_SHA,original_archive=archive,
                        original_phase_receipt=bytes_identity(receipt_raw),original_stage_validation=identity(validation_path),
                        inactive_ledger_identity=identity(ROOT/c.LEDGER_NAME),chunk_bytes=CHUNK_BYTES,
                        transport_root=str(output),part_count=len(rows),parts=rows)
        validate_manifest(manifest,receipt_raw,plan_path.read_bytes(),STAGE)
        manifest_path = output/'transport_manifest.json'
        with manifest_path.open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
        result = dict(status='ready',stage=STAGE,session=SESSION,phase_plan_sha256=PLAN_SHA,original_archive=archive,
                      transport_manifest=dict(identity(manifest_path),path=str(manifest_path)),part_count=len(rows))
        with pointer.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    except BaseException as error:
        with (output/'failure.json').open('x') as stream:json.dump(dict(status='failed_partial_preserved',error=repr(error),original_archive=archive),stream,indent=2)
        raise
    print('AGENT_B_CHUNK_TRANSPORT_RECEIPT',json.dumps(result),flush=True)
    print('AGENT_B_'+STAGE.upper()+'_CHUNK_TRANSPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
