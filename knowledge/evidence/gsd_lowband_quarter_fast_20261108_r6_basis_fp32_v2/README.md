# Explicit physical-basis validator v2

This post hoc comparator amendment preserves the v1 rejection and original
FP64 diagnostic while replaying the producer dtype and object order before
applying the numeric1e-6 threshold. Sampling, gradients, cohort and alpha do
not change. Native repeat failures remain unresolved.

CPU-only code/tests and the actual32-object regression fixture are retained
here. Run `python -m pytest knowledge/evidence/gsd_lowband_quarter_fast_20261108_r6_basis_fp32_v2/test_basis_validation_v2.py`.
The qualification summary binds90spectral+30control cases and2cache archives.
It is not a deployment, ACK or completed617 result.
