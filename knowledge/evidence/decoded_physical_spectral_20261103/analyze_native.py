"""Independent NumPy reconstruction of a sealed decoder FD witness.

Run with the project's NumPy1 CPU environment and one native-diagnostic ZIP.
"""
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.analyze_scan_trajectory import Bundle


def analyze(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        assert len(members) < 20000 and sum(m.file_size for m in members) < 2*1024**3
        assert all(stat.S_IFMT(m.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR)
                   and m.file_size < 256*1024**2 for m in members)
    bundle = Bundle(path)
    try:
        case = 'batches/gaussian/batch0000/'
        report = bundle.json(case + 'decoder_native_audit.json')
        with np.load(io.BytesIO(bundle.read(case + 'decoder_native_audit.npz')),
                     allow_pickle=False) as saved:
            arrays = {key: saved[key].astype(np.float64) for key in saved.files}
        reference, filters = arrays['reference_xyz'], arrays['filters_first_two']
        assert reference.shape == (2, 2048, 3) and filters.shape == (2, 2048, 2048)
        assert all(np.isfinite(value).all() for value in arrays.values())
        denominator = 3*reference.shape[1]
        def loss(points):
            residual = points-reference
            return float(np.einsum('bni,bni->', residual, filters@residual)/denominator)
        baseline, repeat = loss(arrays['baseline_decoded']), loss(arrays['repeat_decoded'])
        assert abs(baseline-report['baseline_loss']) < 1e-7
        assert abs(repeat-report['repeat_loss']) < 1e-7
        gradient, direction = arrays['gradient'], arrays['direction']
        assert np.count_nonzero(direction[1]) == 0
        assert abs(np.linalg.norm(direction[0])-1) < 1e-6
        autodiff = float((gradient*direction).sum())
        comparisons = []
        for index, row in enumerate(report['finite_differences']):
            plus, minus = loss(arrays['fd_%d_plus_decoded' % index]), loss(arrays['fd_%d_minus_decoded' % index])
            assert abs(plus-row['plus_loss']) < 1e-7 and abs(minus-row['minus_loss']) < 1e-7
            finite_difference = (plus-minus)/(2*row['epsilon'])
            tolerance = 1e-5+.1*abs(autodiff)
            assert abs(autodiff) > 1e-5 and abs(finite_difference) > 1e-5
            assert abs(finite_difference-autodiff) <= tolerance
            assert abs(repeat-baseline)/row['epsilon'] <= tolerance
            comparisons.append(dict(scale=row['scale'], independent_fd=finite_difference,
                autograd_directional=autodiff, relative_error=abs(finite_difference-autodiff)/abs(autodiff)))
        assert len(comparisons) == 2 and report['passed']
        assert all(report[key] for key in ('rng_unchanged', 'inputs_unchanged', 'buffers_unchanged'))
        exact = gradient.reshape(2, 2048, 4)
        surrogate = np.zeros_like(exact)
        surrogate[:, :, :3] = 2*(filters@(arrays['baseline_decoded']-reference))/denominator
        full, cheap = exact.reshape(2, -1), surrogate.reshape(2, -1)
        full_norm, cheap_norm = np.linalg.norm(full, axis=1), np.linalg.norm(cheap, axis=1)
        assert (full_norm > 0).all() and (cheap_norm > 0).all()
        return dict(status='independent_numpy_FD_passed', source_archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            fd=comparisons, baseline_loss=baseline, repeat_loss_drift=abs(repeat-baseline),
            residual_freeze_surrogate=dict(local_autograd_cosine_per_object=((full*cheap).sum(1)/(full_norm*cheap_norm)).tolist(),
                surrogate_over_autograd_norm_per_object=(cheap_norm/full_norm).tolist(),
                omitted_residual_jacobian_over_autograd_norm_per_object=(np.linalg.norm(full-cheap, axis=1)/full_norm).tolist()),
            scope='Local decoder/frozen graph; native autograd comparator, not a whole-sampler or accuracy certificate')
    finally:
        bundle.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: analyze_native.py NATIVE_DIAGNOSTIC.zip')
    print(json.dumps(analyze(sys.argv[1]), indent=2, allow_nan=False))
