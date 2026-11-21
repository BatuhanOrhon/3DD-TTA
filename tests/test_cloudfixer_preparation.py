import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import os
import shutil
import subprocess
import sys

import pytest


def test_official_import_patch_is_the_only_author_source_change():
    from scripts.prepare_cloudfixer_transfer import repair_import
    original = '# from pointnet2_ops import pointnet2_utils\n'
    assert repair_import(original) == 'from pointnet2_ops import pointnet2_utils\n'
    with pytest.raises(ValueError):
        repair_import('different source\n')


def test_overlay_manifest_binds_actual_sampler_worker_and_copied_author_sources(tmp_path):
    from scripts.prepare_cloudfixer_transfer import manifest
    official = {'diffusion/transformer.py':'1'*64,'utils/misc.py':'2'*64}
    value = manifest(official)
    assert len(value['dataset_files']) == 16
    assert value['registration']['development_ids'] != value['registration']['confirmation_ids']
    for name, digest in value['overlay_files'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert value['official_files'] == official


def test_packaged_worker_runs_without_any_other_repository_on_pythonpath(tmp_path):
    from scripts.prepare_cloudfixer_transfer import OVERLAY
    for name in OVERLAY:
        target=tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(name,target)
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    result=subprocess.run([sys.executable,'-S',str(tmp_path/'scripts/run_cloudfixer_transfer.py'),'--help'],
                          cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_generated_colab_phases_and_embedded_workers_are_python38_compatible(tmp_path):
    from scripts.prepare_cloudfixer_transfer import colab_phases
    colab_phases(tmp_path)
    assert {p.name for p in tmp_path.glob('*.py')} == {
        'prepare_drive_mount.py','run_smoke.py','run_profile.py','run_development.py'}
    for p in tmp_path.glob('*.py'):
        tree=ast.parse(p.read_text(),feature_version=(3,8))
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('import '):
                ast.parse(node.value,feature_version=(3,8))


def test_generated_phase_exit_receipt_is_readable_json(tmp_path):
    from scripts.prepare_cloudfixer_transfer import colab_phases
    colab_phases(tmp_path)
    for phase in ('smoke','profile','development'):
        tree=ast.parse((tmp_path/('run_'+phase+'.py')).read_text())
        write=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                   and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text')
        eval(compile(ast.Expression(write),'<generated exit receipt>','eval'),
             dict(output=tmp_path,status=0,json=json))
        assert json.loads((tmp_path/'exit_status.json').read_text()) == {'returncode':0}


def test_generated_campaign_uses_only_mounted_drive(tmp_path):
    from scripts.prepare_cloudfixer_transfer import colab_phases
    colab_phases(tmp_path)
    for p in tmp_path.glob('*.py'):
        source=p.read_text()
        assert 'drive_service' not in source and 'authenticate_user' not in source
        assert 'drive_file_id' not in source
    assert 'MountStore' in (tmp_path/'prepare_drive_mount.py').read_text()
    assert '/content/drive/MyDrive/thesis/result' in (tmp_path/'prepare_drive_mount.py').read_text()
