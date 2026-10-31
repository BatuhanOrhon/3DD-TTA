"""Verify immutable phase archive and its complete file manifest before extraction."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile


def extract(archive, receipt, destination):
    archive,receipt,destination=Path(archive),Path(receipt),Path(destination)
    expected=json.loads(receipt.read_text())['archive']
    digest=hashlib.sha256()
    with archive.open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):digest.update(data)
    assert archive.stat().st_size==expected['bytes'] and digest.hexdigest()==expected['sha256']
    assert not destination.exists()
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();names=[i.filename for i in entries]
        assert len(names)==len(set(names))<20000 and sum(i.file_size for i in entries)<2*1024**3
        for info in entries:
            p=PurePosixPath(info.filename)
            assert not p.is_absolute() and '..' not in p.parts and '\\' not in str(p)
            assert stat.S_IFMT(info.external_attr>>16) in (0,stat.S_IFREG)
        assert z.testzip() is None
        manifest=json.loads(z.read('evidence_manifest.json'))
        assert set(names)==set(manifest)|{'evidence_manifest.json'}
        for name,row in manifest.items():
            data=z.read(name)
            assert row==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),name
        destination.mkdir()
        z.extractall(destination)
    print(json.dumps(dict(status='passed',archive_sha256=digest.hexdigest(),files=len(manifest),destination=str(destination))))


if __name__=='__main__':extract(*sys.argv[1:])
