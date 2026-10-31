#!/usr/bin/env python3
"""Thesis baseline: frozen N8 Chamfer + equal probability mean, four source heads."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_chamfer_depth as depth


def configurations(argv=None) -> list[dict]:
    """Use the accepted whole policy; reject tuning through the baseline CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    configs = depth.configurations(['--transfer-study', *arguments])
    for config in configs:
        config.update(method=depth.BASELINE_METHOD, operational_baseline=depth.BASELINE_ID,
                      csv_policy='probability_mean',
                      display_name='Thesis baseline N8 Chamfer + probability mean + ' + config['backbone'],
                      prediction_contract='prediction and CSV use probability_mean; terminal_prediction and logits retain terminal diagnostics',
                      baseline_evidence='knowledge/chamfer_depth_transfer_results_20261031.md')
        depth.validate_transfer_config(config)
    return configs


def worker(directories: list) -> None:
    """Require baseline identity even when replaying saved worker bundles."""
    if not directories or any(json.loads((Path(path) / 'config.json').read_text()).get('operational_baseline')
                              != depth.BASELINE_ID for path in directories):
        raise ValueError('Baseline worker requires the frozen operational baseline profile')
    depth.worker(directories)


def main(argv=None) -> None:
    depth.shared.main(argv, configuration_factory=configurations,
                      worker_entrypoint=__file__, worker_fn=worker)


if __name__ == '__main__':
    main()
