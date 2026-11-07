# Conditional independent dataset/source confirmation — 2026-11-07

[Decision/Code/Open] CPU preparation only, while the frozen
[full ModelNet campaign](gsd_lowband_full_study_20261107.md) runs. No external
GPU execution, calibration or accuracy result. User authorizes this stage only
if full ModelNet shows visible gain; gate already frozen >=1pp M100 PM versus
operational SCD and paired95%objectCI lower>0. Native attribution caveat stays.

Accepted external dataset: ScanObjectNN-C BFTT3D581 `main_split`, all15/s5.
Archive488,341,020bytes/SHAe236151c9f0ad805f59d3571a956b75ada2e1208b030f56f84546cc993b38eb2;
all clean points exactly belong to matching official main_split H5 objects,
label order verified by accepted membership audit. This is a separate dataset,
not a globally unseen benchmark or established GSDTTA corruption-seed parity.
Invalid Oct17 Gaussian-only Drive receipt is excluded.

Separate15class Scan-trained PointMAE checkpoint351,525,407bytes/
SHA53f6c2e46d3c982c9c144a9c822acf5852ef626e2e3124ec0d165c64b9535dac;
Drive `MyDrive/thesis/weights/scanobject_jt.pth`. LION weights stay pinned;
source-model independence here means a separately trained classifier.
Source YAML `cfgs/tta_scanobj.yaml`
SHAbb0f6adecbf60b7ae5e81efe4303c2344bec2518d832580a1d82d9489cac92a4.
The current full-study ASSETS already declare accepted Scan arrays/sourceYAML,
but do not contain the ScanPointMAE checkpoint or a new accepted setup receipt.
Actual checkpoint identity/strictload and separate hashed Scan receipt must pass
before any externalGPU. Retain canonical10cell environment proof separately.

[Decision] Freeze label-free calibration IDs now:39,146,201,273, selected by
sorted NumPyRandomState(20261107).choice(581,4,replace=False), without labels.
Gaussian+Background,seed0,B2; SCD1/GSD0 live hard100 probes, three cleanPM
checkpoints2/4/6 or9/18/27. Median of24finite active perobject actual
gamma.03/eta.003 weighted update-norm ratios becomes externalalpha100, frozen
before accuracy. No clean geometry or labels enter calibration; include these
objects in full581 reports and disclose calibration exposure.

[Open] CPU opt-in dataset boundary and true final singleton support under
review. Preserve ModelNet defaults/even-count guards and frozen remote source.
If ModelNet gate passes: own separate source/asset/operator root; actual
accepted Scan receipt/strictload; B2 repeat/FD checks plus true singleton580
through preparation, native decoder audit and all3arms; then calibration and
full581/all15/s5/seed0/B2+tail1, same3arms. Nine64-ID shards plus final5:
10cachejobs+30inference,4365actual B2/B1 cases,8715predictions per arm.
Preserve all failures, use ceil(count/2), no padding/dropped object or
outcome-dependent subset. Estimate external wholephase only after measured
technical timings; no new source-only run is required by this boundary.

## CPU boundary acceptance

[Code/CPU] Five-file opt-in boundary complete: driver dataset/config/selection/
actualbatch/headcounts; native decoder audit B1/B2 guards with unchanged gates;
three affected testfiles. FullCPU822passed/2skipped/541subtests/7warnings50.91s,
affected94passed/146subtests5.00s; independent focused review35tests/72subtests.
Python3.8syntax/diffcheck pass; no concrete reviewer blocker. Mockworker verifies
IDs578,579,580 as2+1,3/3count,15class arrays; invalid/mismatched receipt rejects.
Executed ModelNet source is unchanged remote snapshot, not this new local code.
Actual Scan checkpoint strictload/nativeB1/repeat/FD and all inference remain
UNRUN and required after the ModelNet gate. CPU support is not transfer evidence.
