#!/usr/bin/env python3
"""Registered Point-MAE N8/B35 SCD-h and independently scaled spectral-z study."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import eval_chamfer_depth as depth
from scripts.scdh_spectralz_study import MODES,RHOS

shared = depth.shared
METHOD = '3dd_scdh_spectralz_study'
LOCAL_STEPS = (.01, .03, .05)
GRAPH = dict(builder='graph_spectral_v2_offdiag.py', isolate_policy='offdiag',
             k=10, delta=.1, gamma=.6, use_4d_gft=False, dynamic=False)
STAGES = dict(technical=32, calibration=32, development=64, confirmation=128, timing=32)


def calibrated_step(path, expected_sha256, modes, rho, *, raw=None):
    raw = Path(path).read_bytes() if raw is None else raw
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Spectral calibration report identity mismatch')
    report = json.loads(raw)
    coefficient = report.get('coefficients', {}).get(str(modes), {})
    b = coefficient.get('b_M')
    if (report.get('schema_version') != 1 or report.get('status') != 'frozen'
            or coefficient.get('status') != 'available' or type(b) not in (int, float)
            or not math.isfinite(b) or b <= 0 or coefficient.get('n_valid', 0) < 1):
        raise ValueError('Require frozen positive M-specific label-free calibration')
    step = rho * b
    if not math.isfinite(step):
        raise ValueError('Nonfinite calibrated spectral-z step')
    return step, dict(path=str(Path(path).absolute()), identity=dict(bytes=len(raw),sha256=expected_sha256), report=report)


def configurations(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-root','label-path','asset-root','result-root','index-manifest','paired-state-root',
                'pointmae-config','pointmae-checkpoint'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--asset-manifest-sha256')
    parser.add_argument('--drive-root')
    parser.add_argument('--stage',choices=tuple(STAGES),required=True)
    parser.add_argument('--arm',choices=('scd','off','spectral','calibration'),required=True)
    parser.add_argument('--alpha-h',type=float,default=.03)
    parser.add_argument('--modes',type=int,choices=MODES)
    parser.add_argument('--rho',type=float,choices=RHOS)
    parser.add_argument('--calibration-report')
    parser.add_argument('--calibration-report-sha256')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(argv)
    if args.alpha_h not in LOCAL_STEPS or args.seed not in ((0,1,2) if args.stage=='confirmation' else (0,)):
        raise ValueError('Unregistered local-step/seed combination')
    if (args.arm=='calibration') != (args.stage=='calibration'):
        raise ValueError('Calibration arm requires its separate label-free stage')
    kappa, calibration = 0., None
    if args.arm=='spectral':
        if args.modes not in MODES or args.rho not in RHOS:
            raise ValueError('Spectral routing needs registered M/rho')
        if args.stage=='technical':
            if args.modes!=800 or args.rho!=1. or args.calibration_report:
                raise ValueError('Native technical gate uses only uncalibrated M800/rho1')
            kappa=0.  # Technical FD observes the gradient before label-free scaling exists.
        elif not (args.calibration_report and args.calibration_report_sha256):
            raise ValueError('Scientific spectral jobs require frozen calibration report and SHA')
        else:
            kappa,calibration=calibrated_step(args.calibration_report,args.calibration_report_sha256,args.modes,args.rho)
    elif any(v is not None for v in (args.modes,args.rho,args.calibration_report,args.calibration_report_sha256)):
        raise ValueError('Control/calibration arms do not accept candidate spectral parameters')
    translated=[]
    for key in ('dataset_root','label_path','asset_root','result_root','index_manifest','paired_state_root',
                'pointmae_config','pointmae_checkpoint','asset_manifest_sha256','drive_root','seed'):
        value=getattr(args,key)
        if value is not None: translated.extend(['--'+key.replace('_','-'),str(value)])
    translated.extend(['--normal-reverse-steps','8','--gamma',str(args.alpha_h),'--eta',str(.003 if args.arm in ('scd','calibration') else 0.),'--p','.95'])
    if args.preview: translated.append('--preview')
    config,=depth.configurations(translated)
    scope=(['gaussian','background','rotation','lidar'] if args.stage=='calibration'
           else ['gaussian','background'] if args.stage in ('technical','timing') else list(shared.source.CORRUPTIONS))
    config.update(method=METHOD,corruptions=scope,csv_policy='probability_mean',
        display_name='SCD-h / spectral-z '+args.arm,graph=GRAPH.copy() if args.arm in ('spectral','calibration') else None,
        scdh_spectralz=dict(stage=args.stage,arm=args.arm,alpha_h_local=args.alpha_h,
            style_scd_step=.003 if args.arm in ('scd','calibration') else 0.,
            kappa_z_spectral_effective=kappa,M_max=args.modes,rho=args.rho,beta=2.,
            spectral_weight=1.,profile='offdiag_static_raw_lambda_truncated_l2',global_diffusion=False,
            local_anchor='initial h0XYZ',graph_anchor='initial h0XYZ',decoder_style='original z0',
            final_style_update_consumed_by_decode=False),
        scdh_technical_gradient_only=args.stage=='technical' and args.arm=='spectral',
        scdh_calibration=calibration,
        comparison_scope='registered SCD-h/spectral-z finite routing search, not full GSDTTA',
        timing_policy='shared original-host reconstruction including actual encoder/graph or cache IO; fresh timing uses separate empty caches')
    validate_config(config)
    return [config]


def validate_config(config, *, calibration_raw=None):
    route=config.get('scdh_spectralz',{})
    arm,stage=route.get('arm'),route.get('stage')
    if (config.get('method')!=METHOD or stage not in STAGES or arm not in ('scd','off','spectral','calibration')
            or config.get('dataset')!='modelnet40_c' or config.get('backbone')!='pointmae'
            or config.get('num_classes')!=40 or config.get('n_examples')!=2468
            or config.get('evaluation_n_examples')!=STAGES[stage]
            or route.get('alpha_h_local') not in LOCAL_STEPS or route.get('beta')!=2.
            or route.get('spectral_weight')!=1. or route.get('global_diffusion') is not False
            or route.get('profile')!='offdiag_static_raw_lambda_truncated_l2'
            or config.get('csv_policy')!='probability_mean' or config.get('batch_size')!=32
            or config.get('normal_reverse_steps')!=8 or config.get('background_reverse_steps')!=35
            or config.get('decoder_style')!='original' or config.get('scheduler_final_alpha_to_one') is not True
            or config.get('lion_ema') is not False or not config.get('paired_state_root')
            or config.get('shape_latent') or config.get('chamfer_experiment') or config.get('semantic_guidance')):
        raise ValueError('Invalid registered SCD-h/spectral-z host or scope')
    scope=(['gaussian','background','rotation','lidar'] if stage=='calibration'
           else ['gaussian','background'] if stage in ('technical','timing') else list(shared.source.CORRUPTIONS))
    if (config.get('corruptions')!=scope or config.get('severity')!=5 or config.get('max_batches')!=0
            or config.get('primary_policy')!='probability_mean' or config.get('trajectory_observation') is not True
            or config.get('chamfer_depth_study') is not True
            or route.get('local_anchor')!='initial h0XYZ' or route.get('graph_anchor')!='initial h0XYZ'
            or route.get('decoder_style')!='original z0' or route.get('final_style_update_consumed_by_decode') is not False
            or (arm=='calibration')!=(stage=='calibration')):
        raise ValueError('Changed registered corruption/route/readout scope')
    protocol=dict(gamma=route['alpha_h_local'],eta=.003 if arm in ('scd','calibration') else 0.,p=.95,total=100,scd_normalize=False)
    if config.get('guidance_protocol')!=protocol or config.get('graph')!=(GRAPH if arm in ('spectral','calibration') else None):
        raise ValueError('Changed route/graph protocol')
    expected=([2,4,6,8],[9,18,27,35])
    if tuple(config['trajectory_selection'][s+'_loop_counts'] for s in ('normal','background'))!=expected:
        raise ValueError('Changed original-z probability-mean horizon/checkpoints')
    selection=config['trajectory_selection']
    expected_selection=dict(normal_candidate_names=['clean_02','clean_04','clean_06','terminal'],
        background_candidate_names=['clean_09','clean_18','clean_27','terminal'],temperature=1.,
        labels_used=False,early_stopping=False, candidate_weighting='equal probability per unique clean checkpoint and terminal',
        style='original encoded shape_latent for every clean and terminal decode')
    if any(selection.get(k)!=v for k,v in expected_selection.items()):
        raise ValueError('Changed frozen probability-mean readout')
    manifest=config.get('selection_manifest',{})
    ids=config.get('selected_original_ids',[])
    if (len(ids)!=STAGES[stage] or any(type(i) is not int or not 0<=i<2468 for i in ids)
            or ids!=sorted(set(ids)) or manifest.get('indices')!=ids
            or manifest.get('dataset')!='modelnet40_c' or manifest.get('n_examples')!=2468
            or manifest.get('stage')!=('heldout' if stage=='confirmation' else 'development')):
        raise ValueError('Changed original-ID subset')
    if config.get('seed') not in ((0,1,2) if stage=='confirmation' else (0,)):
        raise ValueError('Unregistered seed')
    if arm=='spectral':
        if route.get('M_max') not in MODES or route.get('rho') not in RHOS:
            raise ValueError('Unregistered spectral candidate')
        reference=config.get('scdh_calibration')
        if stage=='technical':
            if route['M_max']!=800 or route['rho']!=1. or reference:
                raise ValueError('Changed native technical candidate')
            expected_step=0.
        else:
            if not reference: raise ValueError('Missing frozen spectral calibration')
            expected_step,verified=calibrated_step(reference['path'],reference['identity']['sha256'],route['M_max'],route['rho'],raw=calibration_raw)
            if reference!=verified:raise ValueError('Inline calibration differs from physical frozen report')
        if route.get('kappa_z_spectral_effective')!=expected_step:
            raise ValueError('Spectral-z step differs from pinned calibration')
    elif any(route.get(k) is not None for k in ('M_max','rho')) or route.get('kappa_z_spectral_effective')!=0. or config.get('scdh_calibration'):
        raise ValueError('Control/calibration has candidate spectral updates')
    if route.get('style_scd_step')!=protocol['eta']:
        raise ValueError('Changed semantic SCD style step')
    if config.get('scdh_technical_gradient_only')!=(stage=='technical' and arm=='spectral'):
        raise ValueError('Changed native gradient-only scope')


def worker(directories):
    import shared_trajectory
    if len(directories)!=1: raise ValueError('Exactly one frozen Point-MAE bundle required')
    config=json.loads((Path(directories[0])/'config.json').read_text())
    validate_config(config)
    actual=[]
    def runtime_loader(common,manifest):
        from scdh_spectralz import SpectralStyleRoute, SpectralCalibrationRoute
        runtime=shared.load_guidance_runtime(dict(common,method=shared.ORIGINAL_METHOD),manifest)
        from graph_spectral_v2_offdiag import GraphSpectralDNA
        import scripts.analyze_legacy_trajectory
        if common.get('scdh_calibration'):
            import shutil
            shutil.copyfile(common['scdh_calibration']['path'],Path(directories[0])/'spectral_calibration.json')
        actual.append(common)
        return runtime
    def geometry(raw,dataset,corruption,lion,graph,reconstruct,ops,**kwargs):
        from scdh_spectralz import SpectralStyleRoute, SpectralCalibrationRoute
        from graph_spectral_v2_offdiag import GraphSpectralDNA
        current=actual[0];route=current['scdh_spectralz'];metrics={}
        reference=current.get('scdh_calibration')
        if reference and any(reference['report']['binding'].get(key)!=current.get(key) for key in
                ('asset_manifest_sha256','source_snapshot_sha256','lion_state_before','native_extensions','executed_source_files')):
            raise ValueError('Calibration source/assets differ from actual runtime')
        def sample(**parameters):
            if route['arm'] in ('spectral','calibration'):
                graph=GraphSpectralDNA(k=10,delta=.1,gamma=.6,use_4d_gft=False,device='cuda')
                cache=Path(kwargs['paired_state_path']).with_suffix('.spectral_basis.npz')
                if route['arm']=='calibration':
                    router=SpectralCalibrationRoute(graph, (9,18,27) if corruption=='background' else (2,4,6), cache_path=cache)
                else:
                    router=SpectralStyleRoute(graph,route['M_max'],route['kappa_z_spectral_effective'],cache_path=cache,
                        audit_path=Path(directories[0])/'native_fd'/(corruption+'.npz') if route['stage']=='technical' else None)
                parameters['style_spectral']=router
                result=reconstruct(**parameters)
                metrics.update(router.diagnostics())
                for record in metrics.get('steps',[]):
                    record['alpha_h_local']=route['alpha_h_local']
                    record['applied_local_scd_norm']=[route['alpha_h_local']*n for n in record['local_scd_gradient_norm']]
                return result
            return reconstruct(**parameters)
        kwargs['method']=depth.METHOD
        points,_=depth.reconstruct_geometry(raw,dataset,corruption,lion,None,sample,ops,
            depth_config=current,**kwargs)
        return points,dict(scdh_spectralz=metrics)
    import torch
    try:
        shared.worker(directories,runtime_loader=runtime_loader,geometry_reconstructor=geometry)
    except (FloatingPointError,torch.cuda.OutOfMemoryError) as error:
        if config['scdh_spectralz']['stage']=='development' and config['scdh_spectralz']['arm']=='spectral':
            path=Path(directories[0])/'config.json';failed=json.loads(path.read_text())
            failed.update(scdh_failure_category='numerical_candidate_unavailable',scdh_failure_type=type(error).__name__)
            path.write_text(json.dumps(failed,indent=2,sort_keys=True,allow_nan=False)+'\n')
        raise


def main(argv=None):
    shared.main(argv,configuration_factory=configurations,worker_entrypoint=__file__,worker_fn=worker)


if __name__=='__main__': main()
