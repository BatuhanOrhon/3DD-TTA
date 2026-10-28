"""Bind the future genuine final confirmation ZIP to the reviewed CPU phase template."""
import argparse
import hashlib
from pathlib import Path
import zipfile

parser=argparse.ArgumentParser()
parser.add_argument('--package',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert args.package.is_file() and not args.output.exists()
with zipfile.ZipFile(args.package) as archive:
    assert archive.testzip() is None
raw=args.package.read_bytes()
identity=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
source=(Path(__file__).parent/'accept_confirmation_template.py').read_text()
assert source.count('PACKAGE_IDENTITY = None')==1
source=source.replace('PACKAGE_IDENTITY = None','PACKAGE_IDENTITY = '+repr(identity),1)
with args.output.open('x') as stream:stream.write(source)
print(dict(package=identity,phase_sha256=hashlib.sha256(source.encode()).hexdigest()))
