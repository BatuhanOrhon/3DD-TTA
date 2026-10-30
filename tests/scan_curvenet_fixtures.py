"""Synthetic artifact fixture for the separately registered CurveNet/H scope."""
import hashlib

from scan_curvenet_entropy_report import CHECKPOINT,CORRUPTIONS,KEY,PREPROCESS,PURPOSE,REDUCTION,RULE


def valid_curve_report():
    identity=dict(bytes=1,sha256='a'*64)
    source=dict(bytes=1,sha256=hashlib.sha256(b'x').hexdigest())
    proof=dict(config_identity=identity,asset_manifest_sha256='a'*64,
               executed_source_files={'repo/tta_gsd_v2.py':source},native_extensions={'chamfer':dict(path='/native/chamfer.so',**identity)},
               runtime=dict(cuda='12.1',gpu='synthetic test record'),classifier_state_sha256='b'*64,lion_state={'vae':'c'*64})
    probes=[dict(corruption=corruption,loop_count=count,objects=70,rng_unchanged=True,leaf_grad_fields_unchanged=True,
                 zero_local_finite=True,zero_style_finite=True,zero_local_nonzero=0,zero_style_nonzero=0,
                 zero_decoded_finite=True,zero_decoded_nonzero=0,
                 h_local_min_norm=1.,h_style_min_norm=1.,h_decoded_min_norm=1.)
            for corruption in CORRUPTIONS for count in ((15,23) if corruption=='background' else (5,8))]
    return dict(schema_version=1,purpose=PURPOSE,status='verified_label_free_gpu',dataset='scanobjectnn_c',head='curvenet',
                kind='marginal_entropy',rule=RULE,loss_reduction=REDUCTION,target_semantic_update_ratio=.1,
                coefficients={KEY:.05},ratios={KEY:dict(count=2100,missing=0,median=2.,min=2.,max=2.,p90=2.)},native_probes=probes,
                teacher=dict(checkpoint=CHECKPOINT,checkpoint_identity=identity,state_sha256='b'*64,preprocessing=PREPROCESS,input_layout='bcn'),
                calibration_bundles={'scanobjectnn_c':proof})
