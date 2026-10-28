"""Join sequential normal-CLI downloads with bounded memory and original SHA proof."""
import argparse
import hashlib
import json
from pathlib import Path

from split_stage_archive import BUFFER_BYTES, identity, require, validate_manifest


def local_path(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents), 'Local transport path uses a symlink or traversal')
    return path


def join_archive(manifest,receipt_raw,plan_raw,stage,parts_root,output):
    archive = validate_manifest(manifest,receipt_raw,plan_raw,stage)
    parts_root = local_path(parts_root);output = local_path(output)
    require(parts_root.is_dir(),'Downloaded parts directory is missing')
    aggregate = hashlib.sha256();total = 0
    with output.open('xb') as target:
        for row in manifest['parts']:
            part = local_path(parts_root/row['name'])
            require(part.parent == parts_root and part.is_file() and part.stat().st_size == row['bytes'], 'Missing, foreign or wrong-size downloaded part')
            digest = hashlib.sha256();part_bytes = 0
            with part.open('rb') as stream:
                for block in iter(lambda:stream.read(BUFFER_BYTES),b''):
                    target.write(block);digest.update(block);aggregate.update(block);part_bytes += len(block)
            require(part_bytes == row['bytes'] and digest.hexdigest() == row['sha256'],'Downloaded part SHA differs; failed partial output preserved')
            total += part_bytes
    require(total == archive['bytes'] and aggregate.hexdigest() == archive['sha256'],'Joined bytes differ from original archive; failed output preserved')
    require(identity(output) == {key:archive[key] for key in ('bytes','sha256')},'Persisted joined ZIP differs; failed output preserved')
    return dict(status='verified_original_archive_bytes',stage=stage,bytes=total,sha256=aggregate.hexdigest(),output=str(output),parts=len(manifest['parts']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--phase-receipt',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--stage',choices=('development','confirmation'),required=True)
    parser.add_argument('--parts-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=join_archive(json.loads(args.manifest.read_text()),args.phase_receipt.read_bytes(),args.plan.read_bytes(),args.stage,args.parts_root,args.output)
    print(json.dumps(result,indent=2),flush=True)
    print('AGENT_B_SENSITIVITY_ARCHIVE_JOIN_COMPLETE',flush=True)


if __name__=='__main__':main()
