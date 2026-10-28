"""Reconstruct exact failed-attempt archive bytes without accepting experiment results."""
import argparse
import hashlib
import json
from pathlib import Path

from failed_preservation_phase import require,validate_failed_manifest
from split_stage_archive import BUFFER_BYTES,identity


def safe_local(path):
    path=Path(path).absolute()
    require('..' not in path.parts and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents),'Unsafe local failed-artifact path')
    return path


def join_failed_archive(manifest,receipt_raw,ledger_raw,plan_raw,parts_root,output):
    original=validate_failed_manifest(manifest,receipt_raw,ledger_raw,plan_raw)
    parts_root=safe_local(parts_root);output=safe_local(output)
    digest=hashlib.sha256();size=0
    with output.open('xb') as target:
        for row in manifest['parts']:
            part=safe_local(parts_root/row['name'])
            require(part.parent==parts_root and part.is_file() and part.stat().st_size==row['bytes'],'Missing/foreign/wrong-size failed archive part')
            part_hash=hashlib.sha256();count=0
            with part.open('rb') as stream:
                for block in iter(lambda:stream.read(BUFFER_BYTES),b''):
                    target.write(block);part_hash.update(block);digest.update(block);count+=len(block)
            require(count==row['bytes'] and part_hash.hexdigest()==row['sha256'],'Failed archive part SHA differs; partial output preserved')
            size+=count
    require(size==original['bytes'] and digest.hexdigest()==original['sha256']
            and identity(output)=={key:original[key] for key in ('bytes','sha256')},'Recombined failed archive differs; partial output preserved')
    return dict(status='failed_attempt_archive_bytes_preserved',experimental_result_status='failed',bytes=size,sha256=digest.hexdigest(),output=str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','failed-receipt','failed-ledger','plan','parts-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=join_failed_archive(json.loads(args.manifest.read_text()),args.failed_receipt.read_bytes(),args.failed_ledger.read_bytes(),args.plan.read_bytes(),args.parts_root,args.output)
    print(json.dumps(result,indent=2),flush=True)
    print('AGENT_B_FAILED_CONFIRMATION_ARCHIVE_BYTES_PRESERVED',flush=True)


if __name__=='__main__':main()
