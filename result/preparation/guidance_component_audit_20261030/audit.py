"""CPU-only evidence audit. No model inference, CUDA, or production edits."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = dict(utc=datetime.now(timezone.utc).isoformat(), scope=__doc__,
              torch=torch.__version__, numpy=np.__version__, snapshots={},
              configuration_counts={}, source_bindings_checked=0)
for label, folder, expected in (
    ('fixed', 'guidance_component_study_20261025/assets_v1',
     'f5bca11a543fd5463ef5a09b26cf216b2bd049523d61a97da24e184e5576d0c6'),
    ('sensitivity', 'guidance_component_sensitivity_20261025/assets_v2',
     '73d9fbe16f51826cd13958861d21da81d04ab2c7157131ebd5756f248acb37b8')):
    assets = ROOT / 'result/preparation' / folder
    archive = assets / 'source_snapshot.zip'
    assert sha(archive.read_bytes()) == expected
    sources = {}
    with zipfile.ZipFile(archive) as z:
        for name in ('graph_spectral_v2.py', 'tta_gsd_v2.py',
                     'scripts/eval_multibackbone_guidance.py', 'spectral_guidance_losses.py'):
            member = 'source_snapshot/' + name
            if member not in z.namelist():
                assert label == 'fixed' and name == 'spectral_guidance_losses.py'
                continue
            data = z.read(member)
            assert data == (assets / member).read_bytes()
            sources[name] = sha(data)
    report['snapshots'][label] = dict(zip_sha256=expected, sources=sources)

fixed = ROOT / 'result/preparation/guidance_component_study_20261025'
dev = ROOT / 'result/preparation/guidance_component_sensitivity_20261025/recovery_20261028_r4/local_development_analysis_v1'
for label, folder in [('fixed', fixed / 'development_extracted'), ('sensitivity', dev)]:
    count = 0
    for path in sorted(folder.rglob('config.json')):
        config = json.loads(path.read_text())
        if not config.get('component_study'):
            continue
        count += 1
        assert config['graph']['isolate_policy'] == 'legacy'
        protocol = config['guidance_protocol']
        assert protocol['beta'] == 2 and protocol['M_max'] == 800
        assert protocol['dynamic_graph'] is False
        assert config['decoder_style'] == 'updated'
        assert config['scheduler_final_alpha_to_one'] is False
        # Reused baseline artifacts retain their original v1 identities.
        for source in report['snapshots'][label]['sources']:
            records = [v for k, v in config['executed_source_files'].items()
                       if k.endswith('/' + source)]
            if not records and source == 'spectral_guidance_losses.py':
                assert label == 'sensitivity'
                continue
            allowed = {v['sources'][source] for k, v in report['snapshots'].items()
                       if source in v['sources'] and (label != 'fixed' or k == 'fixed')}
            assert len(records) == 1 and records[0]['sha256'] in allowed, (path, source)
            report['source_bindings_checked'] += 1
        graph_records = [v for k, v in config['executed_source_files'].items()
                         if k.endswith('/graph_spectral_v2.py')]
        assert len(graph_records) == 1
        assert graph_records[0]['sha256'] == report['snapshots']['fixed']['sources']['graph_spectral_v2.py']
    report['configuration_counts'][label] = count
assert report['configuration_counts']['fixed'] == 32
assert report['configuration_counts']['sensitivity'] == 120

with (fixed / 'local_development_analysis/paired_balanced_contrasts.csv').open() as stream:
    report['fixed_s_minus_u'] = [r for r in csv.DictReader(stream)
                                if r['candidate'] == 'spectral' and r['comparator'] == 'unguided']
with (dev / 'balanced_scores.csv').open() as stream:
    report['sensitivity_spectral_scores'] = [r for r in csv.DictReader(stream)
                                            if r['family'] == 'spectral']

# Exercise the archived graph with an explicit Euclidean CPU kNN adapter.
# CUDA tie order and native numerical parity are not asserted by these probes.
captured = {}


class CpuKNN:
    def __init__(self, k, **kwargs):
        self.k = k

    def __call__(self, left, right):
        distance = torch.cdist(left, right, compute_mode='donot_use_mm_for_euclid_dist')
        values, indices = distance.topk(self.k, dim=-1, largest=False)
        captured.update(values=values, indices=indices)
        return values, indices


knn = types.ModuleType('knn_cuda')
knn.KNN = CpuKNN
sys.modules['knn_cuda'] = knn
graph_path = fixed / 'assets_v1/source_snapshot/graph_spectral_v2.py'
graph = load(graph_path, 'archived_graph')
report['graph_probes'] = []
cache_root = fixed / 'paired_cache_extracted/paired_state_v1_seed0'
for dataset in ('modelnet40_c', 'scanobjectnn_c'):
    for corruption in ('gaussian', 'background', 'density_inc'):
        path, = (cache_root / dataset / corruption).glob('*.npz')
        with np.load(path, allow_pickle=False) as z:
            h = torch.from_numpy(z['local_latent'][0:1].copy()).view(1, 2048, 4)
        with torch.no_grad():
            _, basis, eigenvalues = graph.GraphSpectralDNA(device='cpu')(h)
        adjacency = torch.zeros((1, 2048, 2048))
        adjacency.scatter_(2, captured['indices'], torch.exp(-captured['values'].square() / .02))
        adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
        degree = adjacency.sum(-1)
        retained = degree > .6 * degree.mean(-1, keepdim=True)
        adjacency *= retained.unsqueeze(1) * retained.unsqueeze(2)
        degree = adjacency.sum(-1)
        offdiag = adjacency.clone()
        offdiag[:, torch.arange(2048), torch.arange(2048)] = 0
        missed = (degree > 0) & (offdiag.sum(-1) == 0)
        weights = torch.exp(-2 * eigenvalues)
        weights[:, 800:] = 0
        report['graph_probes'].append(dict(
            cache=str(path.relative_to(ROOT)), cache_sha256=sha(path.read_bytes()),
            dataset=dataset, corruption=corruption, selected_batch_position=0,
            retained_self_only_isolates=int(missed.sum()),
            self_only_vertex_indices=missed[0].nonzero().flatten().tolist(),
            self_only_filter_diagonal=(basis[0, missed[0]].square() @ weights[0]).tolist(),
            penalized_zero_row_vertices=int((degree == 0).sum()),
            spectral_weight_mass=float(weights.sum()),
            eigenvalue_799=float(eigenvalues[0, 799]),
            eigenvalue_800=float(eigenvalues[0, 800])))

report['cpu_graph_probe_limit'] = ('First cached object in six predeclared dataset/corruption cells only; '
    'CPU Euclidean kNN substitutes for CUDA. No accuracy or CUDA prevalence inference.')
assert not torch.cuda.is_initialized()
(OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('configuration_counts', 'source_bindings_checked', 'graph_probes')}, indent=2))
