"""Strict Scan/CurveNet/H-only proof; the older two-dataset V1 is independent."""
import math
import re

PURPOSE='scan_curvenet_entropy_calibration'
KEY='semantic_h_curvenet'
RULE='semantic_h_curvenet=0.1/median(CurveNet H/C combined update norm)'
REDUCTION='sum for Chamfer and per-object semantic losses'
CHECKPOINT='checkpoints/3deformrs_curvenet_scanobjectnn_better.t7'
PREPROCESS='raw XYZ -> centroid and L-infinity normalization -> native PointNet2 FPS(1024); transfer from random sampling'
CORRUPTIONS=('uniform','gaussian','background','impulse','upsampling','distortion_rbf','distortion_rbf_inv','density','density_inc','shear','rotation','cutout','distortion','occlusion','lidar')
EPS=1e-12


def require(condition,message):
    if not condition:raise ValueError(message)


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>EPS


def validate_native_probe(probe):
    require(probe.get('rng_unchanged') is True and probe.get('leaf_grad_fields_unchanged') is True,
            'CurveNet native probe changed RNG or .grad fields')
    require(probe.get('zero_local_finite') is True and probe.get('zero_style_finite') is True
            and probe.get('zero_decoded_finite') is True
            and type(probe.get('zero_local_nonzero')) is int and probe['zero_local_nonzero']==0
            and type(probe.get('zero_style_nonzero')) is int and probe['zero_style_nonzero']==0
            and type(probe.get('zero_decoded_nonzero')) is int and probe['zero_decoded_nonzero']==0,
            'CurveNet native zero-weight gradients must be finite and exactly zero')
    require(all(positive(probe.get(name)) for name in ('h_local_min_norm','h_style_min_norm','h_decoded_min_norm')),
            'CurveNet teacher gradients must be finite and nonzero for decoded XYZ/local/style')
    return probe


def validate_report(report):
    require(isinstance(report,dict) and type(report.get('schema_version')) is int and report['schema_version']==1
            and report.get('purpose')==PURPOSE and report.get('status')=='verified_label_free_gpu'
            and report.get('dataset')=='scanobjectnn_c' and report.get('head')=='curvenet'
            and report.get('kind')=='marginal_entropy','Unregistered Scan CurveNet entropy calibration')
    require(report.get('rule')==RULE and report.get('loss_reduction')==REDUCTION
            and report.get('target_semantic_update_ratio')==.1,'CurveNet entropy calibration formula/reduction changed')
    require(isinstance(report.get('coefficients'),dict) and isinstance(report.get('ratios'),dict)
            and set(report['coefficients'])=={KEY} and set(report['ratios'])=={KEY},'H-only calibration candidates changed')
    summary=report['ratios'][KEY];scalar=report['coefficients'][KEY]
    require(isinstance(summary,dict) and type(summary.get('count')) is int and summary['count']==2100 and type(summary.get('missing')) is int
            and summary['missing']==0,'CurveNet H calibration2100-row coverage mismatch')
    require(all(positive(summary.get(name)) for name in ('min','median','p90','max'))
            and summary['min']<=summary['median']<=summary['p90']<=summary['max'] and positive(scalar)
            and math.isclose(scalar,.1/summary['median'],rel_tol=1e-12,abs_tol=0.),'CurveNet H median/coefficient mismatch')
    teacher=report.get('teacher',{})
    require(isinstance(teacher,dict) and isinstance(teacher.get('checkpoint_identity'),dict)
            and teacher.get('checkpoint')==CHECKPOINT and teacher.get('preprocessing')==PREPROCESS
            and teacher.get('input_layout')=='bcn' and re.fullmatch('[0-9a-f]{64}',teacher.get('state_sha256',''))
            and re.fullmatch('[0-9a-f]{64}',teacher.get('checkpoint_identity',{}).get('sha256',''))
            and type(teacher.get('checkpoint_identity',{}).get('bytes')) is int
            and teacher['checkpoint_identity']['bytes']>0,'CurveNet frozen teacher identity/contract missing')
    proofs=report.get('calibration_bundles',{})
    require(isinstance(proofs,dict) and set(proofs)=={'scanobjectnn_c'} and isinstance(proofs['scanobjectnn_c'],dict)
            and proofs['scanobjectnn_c'].get('classifier_state_sha256')==teacher['state_sha256'],
            'CurveNet calibration dataset/teacher state proof mismatch')
    proof=proofs['scanobjectnn_c']
    require(re.fullmatch('[0-9a-f]{64}',proof.get('asset_manifest_sha256',''))
            and re.fullmatch('[0-9a-f]{64}',proof.get('config_identity',{}).get('sha256',''))
            and proof.get('runtime',{}).get('cuda') and proof['runtime'].get('gpu'), 'CurveNet source/asset/GPU proof missing')
    for name in ('executed_source_files','native_extensions'):
        identities=proof.get(name,{})
        require(isinstance(identities,dict) and bool(identities) and all(
            isinstance(value,dict) and type(value.get('bytes')) is int and value['bytes']>=0
            and re.fullmatch('[0-9a-f]{64}',value.get('sha256','')) for value in identities.values()),
            'CurveNet '+name+' identities missing')
    probes=report.get('native_probes',[])
    expected={(corruption,count) for corruption in CORRUPTIONS for count in ((15,23) if corruption=='background' else (5,8))}
    require(isinstance(probes,list) and len(probes)==30 and all(isinstance(probe,dict) for probe in probes)
            and {(probe.get('corruption'),probe.get('loop_count')) for probe in probes}==expected,
            'CurveNet native zero/nonzero probe coverage missing')
    for probe in probes:
        require(type(probe.get('objects')) is int and probe['objects']==70,'CurveNet calibration requires B70 native probes')
        validate_native_probe(probe)
    return report
