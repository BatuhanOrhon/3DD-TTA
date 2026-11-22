"""One immutable, self-verifying Drive file per paired block, including all nine arms."""
import io
import json
import zipfile

from cloudfixer_store import canonical, sha, unit_key


def block_name(unit):
    return unit_key(unit)[:-5] + '.zip'


def restore_or_score(store, unit, available, score, validate):
    name = block_name(unit)
    if name not in available:
        data = score()
        validate(data)
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as archive:
            archive.writestr('records.npz', data)
            archive.writestr('receipt.json', canonical(dict(unit=unit, sha256=sha(data), bytes=len(data))))
        store.put(name, output.getvalue())
    with zipfile.ZipFile(io.BytesIO(store.get(name))) as archive:
        if sorted(archive.namelist()) != ['receipt.json', 'records.npz']:
            raise ValueError('Unexpected checkpoint contents')
        receipt = json.loads(archive.read('receipt.json'))
        data = archive.read('records.npz')
        if receipt != dict(unit=unit, sha256=sha(data), bytes=len(data)):
            raise ValueError('Checkpoint identity or checksum mismatch')
    validate(data)
    return data


def mirror_group(cache, units, target):
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_STORED) as archive:
        for unit in units:
            name = block_name(unit)
            archive.writestr(cache.folder.name + '/' + name, cache.get(name))
    return sha(target.read_bytes())


def import_mirror(root, path):
    """Fill missing verified blocks from the host copy after loss of a DriveFS cache."""
    import re
    from cloudfixer_store import MountStore

    with zipfile.ZipFile(path) as bundle:
        for member in bundle.infolist():
            match = re.fullmatch(r'(background(?:_dev|_smoke|_pm|_follow_(?:smoke|focus|scd|other))?)_([0-9a-f]{16})/(unit_[0-9a-f]{64}\.zip)', member.filename)
            if not match or member.file_size > 8_000_000:
                raise ValueError('Unexpected mirror member')
            folder, name = member.filename.split('/')
            block = bundle.read(member)
            with zipfile.ZipFile(io.BytesIO(block)) as inner:
                data = inner.read('records.npz')
                receipt = json.loads(inner.read('receipt.json'))
                unit = receipt['unit']
                if (unit['campaign'][:16] != match[2] or block_name(unit) != name or
                        receipt != dict(unit=unit, bytes=len(data), sha256=sha(data))):
                    raise ValueError('Invalid mirror receipt')
            store = MountStore(root / folder)
            # Re-serialized ZIP metadata may differ after a replay. Compare the payload.
            if name in store.keys('unit_'):
                def validate(current):
                    if current != data:
                        raise ValueError('Drive and host have different results for this block')
                restore_or_score(store, unit, {name}, None, validate)
            else:
                store.put(name, block)
