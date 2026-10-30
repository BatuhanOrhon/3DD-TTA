#!/usr/bin/env python3
"""Independent sparse-backward Chamfer-only experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_multibackbone_guidance import experiment_main

if __name__ == '__main__':
    experiment_main('sparse')
