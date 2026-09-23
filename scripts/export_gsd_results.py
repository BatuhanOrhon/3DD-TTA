"""Copy immutable GSD ZIP artifacts to a Colab Drive archive."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil


def export_archives(source_root: Path, drive_root: Path, *, stage: str = "all15") -> list[Path]:
    """Export GSD archives, preserving dataset/method/run-id path components."""
    source_root = source_root.resolve()
    drive_root = drive_root.resolve()
    if stage not in {"all15", "pilot", "smoke", "all"}:
        raise ValueError("stage must be all15, pilot, smoke, or all")

    pattern = "*.zip" if stage == "all" else f"*gsd-v1-{stage}-*.zip"
    archives = sorted(
        path for path in source_root.rglob(pattern)
        if path.is_file() and path.parent.name in {"gsd_latent_spectral_v1", "3dd_original"}
    )
    if not archives:
        raise FileNotFoundError(f"No {stage} GSD archives found below {source_root}")

    exported: list[Path] = []
    for archive in archives:
        relative = archive.relative_to(source_root)
        if len(relative.parts) != 3:
            raise ValueError(f"Unexpected artifact layout: {relative}")
        dataset, method, filename = relative.parts
        run_id = Path(filename).stem
        target_dir = drive_root / dataset / method / run_id
        target = target_dir / filename
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if _sha256(target) != _sha256(archive):
                raise FileExistsError(f"Refusing to overwrite different archive: {target}")
            print(f"SKIP existing: {target}")
        else:
            shutil.copy2(archive, target)
            print(f"COPIED {archive} -> {target}")
        exported.append(target)
    return exported


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="./result")
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "all15", "all"), default="all15")
    args = parser.parse_args(argv)
    exported = export_archives(Path(args.source_root), Path(args.drive_root), stage=args.stage)
    print(f"Exported {len(exported)} archive(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
