"""CPU-only readout diagnosis on the previously accepted full Point2Vec archive."""
import csv
import hashlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def recount(raw):
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert sha == "b2f0685cb1f0c3e0a053dff6e19f71503e9b793ba2ed340ff1eb0c6f6081e42d"
    counts = defaultdict(Counter)
    ids = defaultdict(list)
    with ZipFile(raw) as archive:
        for name in archive.namelist():
            if not name.endswith(".npz"):
                continue
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                unit = json.loads(str(data["unit"]))
                corruption = unit["corruption"]
                y = data["labels"].astype(int).reshape(-1)
                assert np.array_equal(unit["ids"], data["ids"]) and unit["seed"] == 2
                ids[corruption].extend(unit["ids"])
                assert np.array_equal(data["proximal_pm_indices"], [20, 10, 0])
                p = data["proximal_pm_probs"]
                assert p.shape == (len(y), 3, 40) and np.isfinite(p).all()
                assert np.all(p >= 0) and np.allclose(p.sum(axis=2), 1)
                logits = data["proximal_pm_logits"].astype(np.float64)
                exp = np.exp(logits - logits.max(axis=2, keepdims=True))
                np.testing.assert_allclose(p, exp / exp.sum(axis=2, keepdims=True), rtol=1e-12, atol=1e-15)
                votes = p.argmax(axis=2)
                eq = p.mean(axis=1).argmax(axis=1) == y
                terminal = votes[:, 2] == y
                n32 = data["proximal_terminal_probs"][:, 0].argmax(axis=1) == y
                cf = data["cloudfixer_probs"][:, 0].argmax(axis=1) == y
                # A single rival strictly above truth at every probe wins under any convex weights.
                truth_p = np.take_along_axis(p, y[:, None, None], axis=2)
                dominated = (p > truth_p).all(axis=1).any(axis=1)
                assert not (dominated & eq).any()
                metrics = dict(t20=votes[:, 0] == y, t10=votes[:, 1] == y,
                               t0=terminal, pm=eq, pm10_0=p[:, 1:].mean(axis=1).argmax(axis=1) == y,
                               pm_repairs_t0=eq & ~terminal, pm_harms_t0=~eq & terminal,
                               common_strict_rival=dominated, cf=cf, n32=n32,
                               dominated_correct_by_cf=dominated & cf,
                               dominated_correct_by_n32=dominated & n32)
                counts[corruption].update({key: int(val.sum()) for key, val in metrics.items()})
    reference = Path(__file__).parents[1] / "cloudfixer_two_colab_20261118/a_full_point2vec_per_corruption.csv"
    with reference.open() as stream:
        accepted = {(r["corruption"], r["method"]): int(r["n_correct"]) for r in csv.DictReader(stream)}
    assert len(counts) == 15
    for c, values in counts.items():
        assert sorted(ids[c]) == list(range(2468))
        for key, policy in [("pm", "proximal_pm"), ("n32", "proximal_terminal"), ("cf", "cloudfixer")]:
            assert values[key] == accepted[c, policy]
        assert values["pm"] - values["t0"] == values["pm_repairs_t0"] - values["pm_harms_t0"]
    total = Counter()
    for values in counts.values():
        total.update(values)
    return dict(raw_sha256=sha, objects_per_corruption=2468, seed=2, all15=dict(total),
                per_corruption=dict(sorted(counts.items())))


if __name__ == "__main__":
    print(json.dumps(recount(Path(sys.argv[1])), indent=2))
