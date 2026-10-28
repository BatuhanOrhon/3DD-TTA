# r4 confirmation cache collection — CPU preparation only

Source: `collect_confirmation_cache.py` (10,576 bytes, SHA256 `694f55a0e138c3d6af34be4f92ef5e44c8b3ac88537ef22aa90efb9e11d15c4f`).

After the confirmation controller and wrapper have actually completed, submit this source through the same sole monitor. Suggested timeout1,800s covers CPU hashing, copying and sealing; the collector performs no GPU work. Exact final marker: `AGENT_B_SENSITIVITY_CONFIRMATION_CACHE_COLLECTION_COMPLETE`.

It requires actual completed confirmation state/validation and inactive completed r4 ledger, current pinned plan/controller, and both physical local/Drive phase ZIP identities. Archived state, plan and freeze must match their current records. It derives all180 expected NPZ paths from registered140 original IDs split into two70-object batches, then joins every final/per-job/trace record. Exactly60 files per seed and30 per dataset/seed are required; original3DD must have no paired-state cache.

Copies are byte-identical into a fresh UTC output directory, using pinned legacy helper89fef… for verification/copy/seal. A fresh owned Drive archive is created. Progress prints after each30 files. Existing caches are never overwritten. Copy/identity or collection failures leave partial output for inspection rather than silently retrying.

Output receipt: `/content/studies/guidance-components/sensitivity_confirmation_cache_evidence_receipt.json`. Download that first, then its physical Drive archive to a fresh local path; verify SHA/CRC and safely extract. Cache paths retain their original `paired_sensitivity_confirmation_v2_seed{0,1,2}/dataset/corruption/hash.npz` hierarchy for the existing local confirmation analyzer. One archive is used; no partitioning or alternate GPU computation was added.

Verification: `PYTHONDONTWRITEBYTECODE=1 /private/tmp/agent_b_numpy1_analysis_env/bin/python test_collect_confirmation_cache.py` passes three tests in0.104s. Fixtures cover both3/4 frozen conditions, exact180 records, incomplete state, missing record, foreign namespace, trace checksum disagreement, Drive escape and parent symlink. Python3.8 AST and whitespace checks pass.

Not executed: any CLI, runtime, GPU or actual confirmation artifact collection. Root must copy the source/test/note into its preparation record and append the supplied dated knowledge note after review.
