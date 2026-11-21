#!/usr/bin/env python3
"""Package the registered rotation sources for deployment after canonical Colab setup."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloudfixer_store import canonical, sha
from scripts.prepare_cloudfixer_transfer import manifest


EXTRA = ('cloudfixer_rotation.py', 'cfgs/cloudfixer_rotation_pilot.json',
         'cfgs/cloudfixer_spectral_rotation_pilot.json', 'cfgs/cloudfixer_scd_rotation_pilot.json',
         'scripts/run_cloudfixer_rotation.py', 'scripts/analyze_cloudfixer_rotation.py',
         'scripts/prepare_cloudfixer_rotation.py', 'scripts/operate_cloudfixer_rotation_colab.py',
         'knowledge/cloudfixer_rotation_pilot_20261121.md',
         'knowledge/cloudfixer_spectral_rotation_pilot_20261121.md', 'knowledge/cloudfixer_scd_rotation_pilot_20261121.md')


def package(author, output):
    from scripts.setup_cloudfixer_point2vec_colab import AUTHOR, AUTHOR_FILES

    author_head = subprocess.check_output(['git','-C',str(author),'rev-parse','HEAD'], text=True).strip()
    if author_head != AUTHOR:
        raise ValueError('Only the pinned author checkpoint source can be packaged')
    output.mkdir(parents=True, exist_ok=False)
    official = output/'official.tar'
    subprocess.run(['git','-C',str(author),'archive','--format=tar','-o',str(official.resolve()),AUTHOR], check=True)
    with tarfile.open(official) as archive:
        files = {m.name: sha(archive.extractfile(m).read()) for m in archive if m.isfile()}
    if sha(json.dumps(files, sort_keys=True).encode()) != AUTHOR_FILES:
        raise ValueError('Unexpected official source identity')
    spec = manifest(files, 'point2vec')
    spec['registration'] = json.loads((ROOT/'cfgs/cloudfixer_rotation_pilot.json').read_text())
    spec['spectral_registration'] = json.loads((ROOT/'cfgs/cloudfixer_spectral_rotation_pilot.json').read_text())
    spec['scd_registration'] = json.loads((ROOT/'cfgs/cloudfixer_scd_rotation_pilot.json').read_text())
    spec['overlay_files'].update({name:sha((ROOT/name).read_bytes()) for name in EXTRA})
    spec['source_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    spec['source_status'] = subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
    with zipfile.ZipFile(output/'research.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in spec['overlay_files']:
            archive.write(ROOT/name, name)
        archive.writestr('manifest.json', canonical(spec))
    receipt = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
               for p in (official, output/'research.zip')}
    (output/'receipt.json').write_bytes(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('author', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.author,args.output),indent=2))


if __name__ == '__main__':
    main()
