import json

import numpy as np
import pytest

from scripts.analyze_cloudfixer_canonical_contributions import canonical_contributions
from scripts.run_cloudfixer_exploration import pack


def test_fixed_rival_strict_harm_repair_tie_and_seed_pooling():
    config = dict(steps=8, start=80, tau=.1, retained=.95)
    times = list(range(80, -1, -10))
    # Two strict reversals, a PM tie won by class 0 and a terminal tie won by class 0.
    # On the strict harm, t30 confidently predicts a third class.
    views = [
        [[.02,.93,.05],[.02,.93,.05],[.01,.02,.97],[.75,.2,.05]],
        [[.02,.93,.05],[.02,.93,.05],[.02,.93,.05],[.75,.2,.05]],
        [[.75,.25,0],[.5,.5,0],[.5,.5,0],[.25,.75,0]],
        [[0,1,0],[0,1,0],[0,1,0],[.5,.5,0]],
        [[1,0,0]]*4,
    ]
    dense = np.zeros((5,9,40), dtype=np.float64)
    dense[:,:,2] = 1
    for slot, native in enumerate([70,50,30,0]):
        dense[:,times.index(native),:3] = np.array(views)[:,slot]
    labels = np.array([0,1,1,0,0])
    records = [dict(unit=dict(config=config, seed=seed),
                    data=pack(dict(probabilities=dense[take], labels=labels[take],
                                   sampler=json.dumps(dict(candidate_indices=times)))))
               for seed,take in [(0,slice(0,3)), (1,slice(3,5))]]
    rows = canonical_contributions(records)
    assert [r['seed'] for r in rows] == [0, 1, 'pooled']
    pooled = rows[-1]
    assert (pooled['n'], pooled['terminal_correct'], pooled['terminal_wrong']) == (5,4,1)
    h, r = pooled['harm'], pooled['repair']
    assert (h['count'],h['strict'],h['tie'],h['pm_tie'],h['terminal_tie']) == (3,1,2,1,1)
    assert (r['count'],r['strict'],r['tie']) == (1,1,0)
    assert [p['native'] for p in h['probes']] == [70,50,30,0]
    assert [p['top1_true'] for p in h['probes']] == [0,0,0,3]
    assert [p['top1_rival'] for p in h['probes']] == [3,3,2,0]
    assert [p['confident_wrong'] for p in h['probes']] == [2,2,2,0]
    assert h['probes'][0]['mean_true_minus_rival'] == pytest.approx((-.91-.5-1)/3)
    assert h['probes'][3]['mean_true_minus_rival'] == pytest.approx((.55+.5+0)/3)
    assert r['probes'][0]['mean_true_minus_rival'] == pytest.approx(.91)
    assert rows[1]['repair']['count'] == 0
    assert rows[1]['repair']['probes'][0]['mean_true_minus_rival'] is None
