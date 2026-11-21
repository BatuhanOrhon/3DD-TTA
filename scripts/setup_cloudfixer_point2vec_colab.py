"""Install the supplemental Point2Vec environment after canonical Colab cell 10.

Run with Colab's base Python, passing the deployed official checkout, package
manifest and classifier checkpoint. Subsequent workers must use
`conda run --no-capture-output -n cf_point2vec python ...`.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path


ENV = 'cf_point2vec'
AUTHOR = 'f043e24414fcf4165fdc892abff21215c9989141'
# Canonical SHA of the name -> SHA256 map from the unmodified author git archive.
AUTHOR_FILES = 'c02dbcf8282d7208101ba17886f5e50ebc259c29f209f84787df02dc32ac2e8a'
CHECKPOINT = '6c1d4884bdfdb3ad42db660ab0d2b1a97deb9fd56d07b438d3179208f7693b4d'
TORCH = ('torch==2.1.2+cu121', 'torchvision==0.16.2+cu121')
PACKAGES = (
    'numpy==1.24.4', 'scipy==1.10.1', 'pandas==2.0.3', 'h5py==3.10.0',
    'scikit-learn==1.3.2', 'pyyaml==6.0.1', 'timm==0.9.16',
    'pytorch-lightning==1.9.5', 'lightning-bolts==0.7.0', 'torchmetrics==0.11.4',
    'fvcore==0.1.5.post20221221', 'iopath==0.1.10', 'ninja==1.11.1.1',
    'setuptools==69.5.1', 'wheel==0.43.0', 'easydict==1.13',
)
NATIVE = {
    'pytorch3d': ('facebookresearch/pytorch3d', '2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7'),
    'chamfer-distance': ('otaheri/chamfer_distance', 'f86f6f7cadd3aca642704573d1626c67ca2e2846'),
}
CONTENT = Path('/content')
STATE = CONTENT / '.cf_point2vec_setup'
SCRIPT = Path(__file__).resolve()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_inputs(official, manifest, checkpoint):
    spec = json.loads(manifest.read_text())
    files = spec['official_files']
    if spec['classifier_sha256'] != CHECKPOINT:
        raise ValueError('Expected the Point2Vec ModelNet40 package manifest')
    if spec['author_commit'] != AUTHOR or hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()).hexdigest() != AUTHOR_FILES:
        raise ValueError('Point2Vec requires the exact unpatched official source manifest')
    for name, expected in files.items():
        if sha(official / name) != expected:
            raise ValueError(f'Official source mismatch: {name}')
    if checkpoint.stat().st_size != 265563493 or sha(checkpoint) != CHECKPOINT:
        raise ValueError(f'Invalid official Point2Vec ModelNet40 checkpoint: {checkpoint}')


def canonical_toolchain():
    receipt = json.loads((CONTENT / 'cli_setup_verification.json').read_text())
    if receipt['status'] != 'passed' or receipt['torch_cuda'] != '12.1':
        raise ValueError('Canonical ten-cell setup must pass first')
    for phase in ('prepare', 'emd', 'chamfer', 'pointnet2', 'pvcnn-smoke'):
        if not (CONTENT / '3DD-TTA/.colab_setup/native' / (phase + '.complete')).is_file():
            raise ValueError(f'Missing canonical completion marker: {phase}')
    query = ("import json, os, shutil, sys, torch; "
             "assert torch.cuda.is_available(); "
             "print(json.dumps([sys.prefix, shutil.which(os.environ['CC']), "
             "shutil.which(os.environ['CXX']), torch.cuda.get_device_name(), "
             "'.'.join(map(str, torch.cuda.get_device_capability()))]))")
    prefix, cc, cxx, gpu, arch = json.loads(subprocess.check_output(
        ['conda', 'run', '--no-capture-output', '-n', '3dd_tta_env', 'python', '-c', query],
        text=True))
    prefix = Path(prefix)
    if gpu != receipt['gpu'] or 'A100' not in gpu or arch != '8.0':
        raise ValueError(f'Unexpected canonical GPU: {gpu} (SM {arch})')
    for compiler in (cc, cxx):
        if not compiler or not Path(compiler).is_relative_to(prefix):
            raise ValueError(f'Expected canonical Conda compiler: {compiler}')
        version = subprocess.check_output([compiler, '-dumpfullversion'], text=True).strip()
        if version != '11.4.0':
            raise ValueError(f'Expected GCC 11.4.0: {compiler} reported {version}')
    if 'release 12.1,' not in subprocess.check_output([prefix/'bin/nvcc', '--version'], text=True):
        raise ValueError('Expected canonical CUDA 12.1 toolkit')
    return prefix, cc, cxx, arch


def cuda_paths(prefix):
    paths = []
    for name in ('include/cuda_runtime_api.h', 'lib/libcudart.so'):
        for root in (prefix / 'targets/x86_64-linux', prefix):
            if (root / name).is_file():
                paths.append(str((root / name).parent))
                break
        else:
            raise FileNotFoundError(f'Missing {name} in canonical toolkit {prefix}')
    return paths


def run(phase, command):
    """Stream both child outputs, including a timestamped last-output heartbeat."""
    print(f'{phase} {" ".join(map(str, command))}', flush=True)
    last = ['started']
    finished = threading.Event()
    start = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors='replace', bufsize=1) as process:
        def progress():
            while not finished.wait(60):
                print(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} '
                      f'phase={phase} pid={process.pid} elapsed={int(time.monotonic()-start)}s '
                      f'last={last[0]}', flush=True)
        thread = threading.Thread(target=progress, daemon=True)
        thread.start()
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                if line.strip():
                    last[0] = line.strip()[-240:]
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            finished.set()
            thread.join()


def stage(state, phase, identity, command):
    marker = state / (phase + '.complete')
    if marker.is_file() and marker.read_text() == identity:
        print(f'{phase} reusing completed install', flush=True)
        return
    marker.unlink(missing_ok=True)
    run(phase, command)
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(identity)
    temporary.replace(marker)


def package_identity():
    import torch
    if sys.version_info[:3] != (3, 10, 14) or torch.version.cuda != '12.1':
        raise RuntimeError(f'Unexpected Python/Torch CUDA: {sys.version}, {torch.__version__}')
    versions = {}
    for requirement in TORCH + PACKAGES:
        name, version = requirement.split('==')
        versions[name] = metadata.version(name)
        if versions[name] != version:
            raise RuntimeError(f'Expected {requirement}, found {versions[name]}')
    sources = {}
    for package, (repo, commit) in NATIVE.items():
        provenance = json.loads(metadata.distribution(package).read_text('direct_url.json'))
        if (provenance['vcs_info']['commit_id'] != commit or
                provenance['vcs_info']['vcs'] != 'git' or provenance['url'] != f'https://github.com/{repo}.git'):
            raise ValueError(f'{package} source mismatch: {provenance}')
        versions[package] = metadata.version(package)
        sources[package] = dict(url=provenance['url'], commit_id=commit)
    return dict(package_versions=versions, native_sources=sources)


def native_extensions():
    import torch  # Load the shared Torch libraries before importing the CUDA extension.
    from pytorch3d import _C
    binary = Path(_C.__file__).resolve(strict=True)
    relative = binary.relative_to(Path(sys.prefix).resolve())
    with binary.open('rb') as stream:
        if binary.suffix != '.so' or stream.read(4) != b'\x7fELF':
            raise ValueError(f'Expected an ELF PyTorch3D extension: {binary}')
    return {relative.as_posix(): sha(binary)}


def runtime_identity() -> dict:
    """Recheck the live worker against setup and expose its build for exact cache recovery."""
    prefix = Path(sys.prefix)
    if prefix.name != ENV or sys.version_info[:3] != (3, 10, 14):
        raise RuntimeError(f'Expected {ENV} Python 3.10.14, found {prefix} {sys.version}')
    receipt = json.loads((STATE / 'verified.complete').read_text())
    identity = receipt['identity']
    if (identity['python_prefix'] != str(prefix) or identity['script_sha256'] != sha(SCRIPT)
            or identity['official'] != AUTHOR_FILES or receipt['classifier_sha256'] != CHECKPOINT):
        raise ValueError('Point2Vec setup receipt differs from current prefix, setup script or author inputs')
    freeze_sha = receipt['pip_freeze_sha256']
    if sha(STATE / 'pip-freeze.txt') != freeze_sha:
        raise ValueError('Saved pip freeze differs from verified setup receipt')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'])
    if hashlib.sha256(freeze).hexdigest() != freeze_sha:
        raise ValueError('Live pip freeze differs from verified setup receipt')
    packages = package_identity()
    extensions = native_extensions()
    if extensions != receipt['native_extensions']:
        raise ValueError('Loaded native extension SHA differs from verified Point2Vec setup receipt')
    return dict(python_prefix=str(prefix), script_sha256=identity['script_sha256'],
                toolchain=identity['toolchain'], official=AUTHOR_FILES, classifier_sha256=CHECKPOINT,
                pip_freeze_sha256=freeze_sha, native_extensions=extensions, **packages)


def smoke(official, checkpoint):
    import torch
    package_identity()
    os.chdir(official)
    sys.path.insert(0, str(official))
    import adapt  # Its ChamferDistance calls the pinned PyTorch3D KNN extension.
    from classifier.point2vec.models import Point2VecClassification
    from classifier.point2vec.utils.checkpoint import extract_model_checkpoint
    from pytorch3d.ops import knn_points, sample_farthest_points
    from pytorch3d import _C

    points = torch.rand(1, 1024, 3, device='cuda', requires_grad=True)
    reference = points.detach().contiguous() + 0.01
    sampled, _ = sample_farthest_points(reference, K=32)
    dist = knn_points(points, sampled, K=4).dists
    left, right, _, _ = adapt.chamfer_dist_fn(points, reference)
    for loss in (dist.mean(), left.mean() + right.mean()):
        gradient, = torch.autograd.grad(loss, points)
        if not torch.isfinite(gradient).all() or not gradient.abs().any():
            raise RuntimeError('Native CUDA distance backward check failed')
    head = Point2VecClassification().cuda()
    head.setup()
    weights = extract_model_checkpoint(str(checkpoint))
    head.load_state_dict(weights, strict=True)
    with torch.inference_mode():
        logits = head.cuda().eval().requires_grad_(False)(points.detach())
    if logits.shape != (1, 40) or not torch.isfinite(logits).all():
        raise RuntimeError('Official classifier CUDA forward failed')
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), state_keys=len(weights),
                          native_pytorch3d=_C.__file__, logits_shape=list(logits.shape),
                          classifier_sha256=CHECKPOINT)), flush=True)
    return native_extensions()


def install(official, manifest, checkpoint):
    check_inputs(official, manifest, checkpoint)
    toolkit, cc, cxx, arch = canonical_toolchain()
    include, lib = cuda_paths(toolkit)
    environments = json.loads(subprocess.check_output(['conda', 'env', 'list', '--json'], text=True))
    matches = [Path(p) for p in environments['envs'] if Path(p).name == ENV]
    if not matches:
        run('python', ['conda', 'create', '-y', '-n', ENV, '--override-channels', '-c',
                       'conda-forge', 'python=3.10.14', 'pip=24.0'])
    conda = ['conda', 'run', '--no-capture-output', '-n', ENV]
    query = 'import sys; assert sys.version_info[:3] == (3,10,14); print(sys.prefix)'
    prefix = Path(subprocess.check_output(conda + ['python', '-c', query], text=True).strip())
    if prefix.name != ENV or prefix.resolve() == toolkit.resolve():
        raise RuntimeError(f'Unexpected supplemental prefix: {prefix}')
    state = prefix / '.cf_point2vec_setup'
    state.mkdir(exist_ok=True)
    torch_lib = prefix / 'lib/python3.10/site-packages/torch/lib'
    cuda_vars = dict(CC=cc, CXX=cxx, CUDAHOSTCXX=cxx, CUDA_HOME=str(toolkit),
                     CPATH=include, LIBRARY_PATH=lib, TORCH_CUDA_ARCH_LIST=arch,
                     LD_LIBRARY_PATH=os.pathsep.join(filter(None, [str(torch_lib), lib,
                             str(prefix/'lib'), os.environ.get('LD_LIBRARY_PATH')])),
                     FORCE_CUDA='1', MAX_JOBS='2', PYTHONNOUSERSITE='1')
    run('toolchain', ['conda', 'env', 'config', 'vars', 'set', '-n', ENV,
                      *(f'{k}={v}' for k, v in cuda_vars.items())])
    identity = json.dumps(dict(script_sha256=sha(SCRIPT), python_prefix=str(prefix),
                               toolchain=cuda_vars, official=AUTHOR_FILES), sort_keys=True)
    pip = conda + ['python', '-m', 'pip', 'install']
    stage(state, 'torch', identity, pip + ['--extra-index-url', 'https://download.pytorch.org/whl/cu121', *TORCH])
    stage(state, 'packages', identity, pip + list(PACKAGES) + list(TORCH))
    for name, (repo, commit) in NATIVE.items():
        url = f'git+https://github.com/{repo}.git@{commit}'
        stage(state, name, identity, pip + ['--no-build-isolation', '--no-deps',
                                            '--force-reinstall', '-v', url])
    run('dependency-check', conda + ['python', '-m', 'pip', 'check'])
    verified = STATE / 'verified.tmp'
    run('verify', conda + ['python', '-u', str(SCRIPT), '--smoke', '--smoke-report', str(verified),
                           '--official-root', str(official),
                           '--manifest', str(manifest), '--classifier-checkpoint', str(checkpoint)])
    receipt = json.loads(verified.read_text())
    receipt.update(identity=json.loads(identity), classifier_sha256=CHECKPOINT)
    verified.write_text(json.dumps(receipt, indent=2) + '\n')
    verified.replace(STATE / 'verified.complete')
    print('CF_POINT2VEC_SETUP_VERIFIED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--classifier-checkpoint', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--smoke-report', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    official, manifest, checkpoint = (p.resolve(strict=True) for p in
        (args.official_root, args.manifest, args.classifier_checkpoint))
    if args.smoke:
        extensions = smoke(official, checkpoint)
        if args.smoke_report:
            frozen = STATE / 'pip-freeze.txt'
            frozen.write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all']))
            args.smoke_report.write_text(json.dumps(dict(native_extensions=extensions, pip_freeze_sha256=sha(frozen))))
        return
    STATE.mkdir(exist_ok=True)
    with (STATE/'install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (STATE/'verified.complete').unlink(missing_ok=True)
        install(official, manifest, checkpoint)


if __name__ == '__main__':
    main()
