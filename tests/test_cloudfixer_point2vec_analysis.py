import io
import stat
import zipfile

import numpy as np
import pytest

from cloudfixer_store import canonical, sha, unit_key
from scripts import analyze_cloudfixer_point2vec as analysis


def payload():
    labels = np.array([1, 0])
    unit = dict(campaign='a'*64, corruption='cutout', seed=2, batch_size=1, offset=0,
                order_sha256=sha(canonical([1, 0])), ids=[1, 0])
    truth = labels[unit['ids']]
    rng = dict(python=[3], numpy=['MT19937'], cpu=[7], cuda=[9])
    data = dict(unit=canonical(unit).decode(), ids=unit['ids'], labels=truth, rng_before_sha256='b'*64,
                rng_after=canonical(rng).decode(), input_sha256=['c'*64, 'd'*64],
                seconds=np.ones((2, 4)), peak_mb=np.full((2, 4), 128.))
    for arm in analysis.ARMS:
        probes = analysis.PROBES.get(arm, [0])
        logits = np.full((2, len(probes), 40), -20., dtype=np.float32)
        logits[:, :, :2] = 0
        if arm == 'proximal_pm':
            # Probability mean selects 0 despite the larger single logit for 1.
            logits[:, :2, 0] = 4
            logits[:, 2, 1] = 12
        else:
            logits[:, :, 1] = 4
        probs = np.exp(logits.astype(float))
        probs /= probs.sum(-1, keepdims=True)
        data.update({arm+'_logits': logits, arm+'_probs': probs,
                     arm+'_entropy': -(probs*np.log(probs)).sum(-1)})
        if arm in analysis.PROBES:
            data[arm+'_indices'] = probes
            data[arm+'_logsnr'] = [10-t/10 for t in probes]
    vote = np.full((2, 40), .01)
    vote[np.arange(2), truth] = .61
    data['vote_log_probs'] = np.log(vote)
    return unit, labels, data


def encode(values):
    buf = io.BytesIO()
    np.savez_compressed(buf, **values)
    return buf.getvalue()


def test_recount_rebuilds_probability_mean_and_distinguishes_plain_cf_from_vote():
    unit, labels, data = payload()
    correct, seconds, peak, vote, after = analysis.recount_block(encode(data), unit, labels, 'b'*64, {})
    np.testing.assert_array_equal(correct, [[False, True, False, False], [True, False, True, True]])
    assert vote == 2 and after == sha(data['rng_after'].encode())
    np.testing.assert_array_equal(seconds, [2]*4)
    np.testing.assert_array_equal(peak, [128]*4)


@pytest.mark.parametrize(('field', 'wrong', 'error'), [
    ('labels', [1, 0], 'label'), ('rng_before_sha256', 'a'*64, 'RNG'),
    ('proximal_pm_entropy', np.zeros((2, 3)), 'entropy'), ('proximal_pm_indices', [30, 10, 0], 'probe'),
    ('proximal_pm_probs', np.full((2, 3, 40), .025), 'probs')])
def test_recount_rejects_inconsistent_block(field, wrong, error):
    unit, labels, data = payload()
    data[field] = wrong
    with pytest.raises(ValueError, match=error):
        analysis.recount_block(encode(data), unit, labels, 'b'*64, {})


def test_permutation_and_final_storage_block():
    order = list(reversed(range(2468)))
    blocks = list(analysis.units_for(order, 'cutout', 'x'))
    assert len(blocks) == 78 and blocks[-1]['offset'] == 2464 and blocks[-1]['ids'] == [3, 2, 1, 0]
    assert blocks[-1]['order_sha256'] == sha(canonical(order))
    order[0] = order[1]
    with pytest.raises(ValueError, match='permutation'):
        list(analysis.units_for(order, 'cutout', 'x'))


def test_receipt_binds_expected_json_to_payload_bytes():
    unit, _, _ = payload()
    data = b'raw'
    name = 'data_'+sha(data)+'.npz'
    receipt = dict(schema=1, unit=unit, payload=name, sha256=sha(data), bytes=len(data))
    for actual, saved, error in [(data, receipt, None), (b'bad', receipt, 'SHA'),
                                  (data, dict(receipt, unit=dict(unit, offset=32)), 'receipt')]:
        with zipfile.ZipFile(io.BytesIO(), 'w') as z:
            z.writestr('raw/'+unit_key(unit), canonical(saved))
            z.writestr('raw/'+name, actual)
            if error:
                with pytest.raises(ValueError, match=error):
                    analysis.read_receipt(z, 'raw/', unit)
            else:
                assert analysis.read_receipt(z, 'raw/', unit)[0] == data


def test_bootstrap_resamples_paired_original_ids_across_corruptions():
    a = np.array([[True, True], [False, False], [True, False]])
    b = np.array([[False, False], [True, True], [False, True]])
    result = analysis.paired(a, b)
    assert result == dict(delta_pp=0., ci95_low_pp=-100., ci95_high_pp=100., repair=3, harm=3)
    # Paired clusters with opposite per-corruption effects cancel within every resample.
    assert analysis.paired(a[[2]], b[[2]])['ci95_high_pp'] == 0.


@pytest.mark.parametrize('name', ['../escape', 'run/./alias', 'run/double//slash', 'C:/outside', 'run\\escape'])
def test_zip_rejects_unsafe_paths(name):
    with zipfile.ZipFile(io.BytesIO(), 'w') as z:
        z.writestr(name, b'')
        with pytest.raises(ValueError, match='path'):
            analysis.members(z)


def test_zip_rejects_symlink_and_duplicate_name():
    with zipfile.ZipFile(io.BytesIO(), 'w') as z:
        link = zipfile.ZipInfo('run/raw')
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(link, b'elsewhere')
        with pytest.raises(ValueError, match='path'):
            analysis.members(z)
    with zipfile.ZipFile(io.BytesIO(), 'w') as z:
        z.writestr('run/config.json', b'one')
        with pytest.warns(UserWarning):
            z.writestr('run/config.json', b'two')
        with pytest.raises(ValueError, match='duplicate'):
            analysis.members(z)


def test_csv_disallows_reported_percentage_and_duplicate_rows():
    summary = [dict(method='source', n_correct=1, accuracy=.5)]
    analysis.compare_csv(b'method,n_correct,accuracy\nsource,1,0.5\n', summary, ['method'])
    for csv_row, error in [('source,1,50\n', 'accuracy'), ('source,1,0.5\nsource,1,0.5\n', 'duplicate')]:
        with pytest.raises(ValueError, match=error):
            analysis.compare_csv(('method,n_correct,accuracy\n'+csv_row).encode(), summary, ['method'])


def test_raw_sha_and_fresh_output_are_checked_before_writes(tmp_path):
    archive, output = tmp_path/'raw.zip', tmp_path/'derived'
    archive.write_bytes(b'input')
    with pytest.raises(ValueError, match='SHA256'):
        analysis.analyze(archive, '0'*64, output)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(FileExistsError):
        analysis.analyze(archive, sha(b'input'), output)
    assert archive.read_bytes() == b'input'
