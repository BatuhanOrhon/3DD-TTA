#!/usr/bin/env python3
"""Label-free Scan/CurveNet H-only calibration, after the original campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from gsd_calibration import scalar_summary
from legacy_trajectory import rng_hash
from research_artifacts import CORRUPTIONS
from scan_curvenet_entropy_report import (CHECKPOINT,EPS,KEY,PREPROCESS,PURPOSE,REDUCTION,RULE,
                                          require,validate_native_probe,validate_report)
from scripts import eval_multibackbone_guidance as runner
from scripts.analyze_scan_trajectory import Bundle
from scripts.calibrate_chamfer_variants import update_gradient_rows
from semantic_trajectory import SemanticTrajectoryObserver


class CurveEntropyCalibrationObserver(SemanticTrajectoryObserver):
    def __init__(self,*args,original_ids,corruption,**kwargs):
        classify=kwargs.pop('classify_semantic_decoded')
        self.decoded_probe=None
        def retain_decoded(decoded):
            self.decoded_probe=decoded
            return classify(decoded)
        super().__init__(*args,classify_semantic_decoded=retain_decoded,**kwargs)
        require(self.guidance_head=='curvenet' and self.kind=='marginal_entropy','CurveNet H-only observer required')
        self.original_ids,self.corruption=list(original_ids),corruption
        self.calibration_rows,self.native_probes=[],[]

    def guidance_components(self,*,ch_loss,semantic_loss,noisy_local,style,loop_count,timestep,eta,gamma,**unused):
        if semantic_loss is None:return
        require(loop_count in self.steps[1:] and self.decoded_probe is not None,'Unexpected CurveNet H probe')
        parameters=(noisy_local,style);before_rng=rng_hash()
        previous=[None if value.grad is None else value.grad.detach().clone() for value in parameters]
        c=torch.autograd.grad(ch_loss,parameters,retain_graph=True)
        h=torch.autograd.grad(semantic_loss,parameters,retain_graph=True)
        zero=torch.autograd.grad(0.*semantic_loss,(*parameters,self.decoded_probe),retain_graph=True)
        decoded,=torch.autograd.grad(semantic_loss,(self.decoded_probe,),retain_graph=True)
        def norms(value):return value.detach().flatten(1).double().norm(dim=1)
        decoded_norms=norms(decoded)
        require(all(torch.isfinite(value).all() for value in (*c,*h,*zero,decoded)),'Nonfinite CurveNet calibration gradient')
        unchanged=all((value.grad is None if saved is None else value.grad is not None and torch.equal(value.grad,saved))
                      for value,saved in zip(parameters,previous))
        proof=dict(corruption=self.corruption,loop_count=loop_count,objects=len(noisy_local),
            rng_unchanged=rng_hash()==before_rng,leaf_grad_fields_unchanged=unchanged,
            zero_local_finite=bool(torch.isfinite(zero[0]).all()),zero_style_finite=bool(torch.isfinite(zero[1]).all()),
            zero_decoded_finite=bool(torch.isfinite(zero[2]).all()),zero_decoded_nonzero=int(torch.count_nonzero(zero[2])),
            zero_local_nonzero=int(torch.count_nonzero(zero[0])),zero_style_nonzero=int(torch.count_nonzero(zero[1])),
            h_local_min_norm=float(norms(h[0]).min()),h_style_min_norm=float(norms(h[1]).min()),
            h_decoded_min_norm=float(decoded_norms.min()))
        validate_native_probe(proof)
        rows=update_gradient_rows(c,h,eta=eta,gamma=gamma)
        require(len(rows)==len(self.original_ids),'CurveNet calibration original-ID mismatch')
        for original_id,row,decoded_norm in zip(self.original_ids,rows,decoded_norms.cpu().tolist()):
            row.update(original_id=int(original_id),corruption=self.corruption,loop_count=loop_count,timestep=int(timestep),
                       teacher_decoded_grad_norm=decoded_norm,batch_raw_h=float(semantic_loss.detach()),eta_local=eta,gamma_style=gamma)
            self.calibration_rows.append(row)
        self.native_probes.append(proof);self.decoded_probe=None

    def diagnostics(self):
        result=super().diagnostics()
        result['curvenet_entropy_calibration']=dict(purpose=PURPOSE,rows=self.calibration_rows,native_probes=self.native_probes,
            gradient_calls=4*len(self.native_probes),labels_used_for_selection=False,weighted_calls=0,kl_calls=0,
            updates='C trajectory; H weight0; diagnostic gradients never assign .grad')
        return result


def validate_run(config):
    require(config['dataset']=='scanobjectnn_c' and config['seed']==0 and config['batch_size']==70
            and config['backbones']==['curvenet'] and config['backbone']=='curvenet','CurveNet calibration requires Scan/seed0/B70/CurveNet only')
    require(config['corruptions']==list(CORRUPTIONS) and config['max_batches']==0
            and config.get('selection_manifest',{}).get('stage')=='development'
            and len(config.get('selected_original_ids',[]))==70,'CurveNet calibration requires development70/all15')
    protocol=config['guidance_protocol'];semantic=config.get('semantic_guidance',{})
    require(protocol['chamfer_experiment']==dict(variant='semantic',semantic_weight=0.)
            and protocol['loss_weights']==dict(chamfer=1.,spectral=0.,invariant=0.) and protocol['skip_inactive_components'] is True
            and protocol['eta']==protocol['gamma']==.01 and protocol['p']==.95 and config['graph'] is None,'CurveNet calibration must use unchanged C trajectory')
    require(semantic.get('head')=='curvenet' and semantic.get('kind')=='marginal_entropy' and semantic.get('loss_reduction')=='sum'
            and semantic.get('teacher_checkpoint')==CHECKPOINT and semantic.get('teacher_preprocessing')==PREPROCESS
            and semantic.get('teacher_input_layout')=='bcn','CurveNet teacher contract mismatch')


def coefficient_report(rows,ids):
    require(len(ids)==70 and ids==sorted(set(ids)),'Expected frozen sorted development70 IDs')
    expected={(corruption,index,count) for corruption in CORRUPTIONS for index in ids
              for count in ((15,23) if corruption=='background' else (5,8))}
    require(len(rows)==len(expected) and {(row['corruption'],row['original_id'],row['loop_count']) for row in rows}==expected,
            'CurveNet H calibration coverage missing or duplicated')
    for row in rows:
        require(all(type(row.get(name)) in (int,float) and math.isfinite(row[name]) and row[name]>EPS
                    for name in ('chamfer_update_norm','candidate_update_norm','combined_update_ratio',
                                 'local_candidate_grad_norm','style_candidate_grad_norm','teacher_decoded_grad_norm')),
                'CurveNet H calibration requires finite nonzero gradients/update ratios')
        require(math.isclose(row['combined_update_ratio'],row['candidate_update_norm']/row['chamfer_update_norm'],rel_tol=1e-10),
                'CurveNet H update-norm ratio mismatch')
    summary=scalar_summary(row['combined_update_ratio'] for row in rows)
    return dict(schema_version=1,purpose=PURPOSE,dataset='scanobjectnn_c',head='curvenet',kind='marginal_entropy',
        rule=RULE,loss_reduction=REDUCTION,target_semantic_update_ratio=.1,coefficients={KEY:.1/summary['median']},ratios={KEY:summary},
        update_rate_mapping='local=eta, style=gamma',selection='gradient-only; no label/accuracy coefficient fitting',
        local_cosine=scalar_summary(row['local_cosine'] for row in rows),style_cosine=scalar_summary(row['style_cosine'] for row in rows))


def summarize_bundle(path):
    bundle=Bundle(path)
    try:
        config=bundle.json('config.json');validate_run(config)
        require(config.get('status')=='partial' and config.get('completed_batches')=={name:1 for name in CORRUPTIONS},'Incomplete declared CurveNet development subset')
        require(config.get('curvenet_entropy_calibration',{}).get('purpose')==PURPOSE
                and config['classifier_state_before_sha256']==config['classifier_state_after_sha256']
                and config['lion_state_before']==config['lion_state_after'],'CurveNet calibration model state/purpose mismatch')
        require(config.get('runtime',{}).get('cuda') and config['runtime'].get('gpu') and config.get('native_extensions'),
                'CurveNet native GPU evidence missing')
        sealed=bundle.json('bundle_manifest.json');sources=config['executed_source_files']
        require(all('repo/'+name in sources for name in ('tta_gsd_v2.py','semantic_trajectory.py','scripts/calibrate_scan_curvenet_entropy.py','scan_curvenet_entropy_report.py')),
                'CurveNet calibration executed source proof missing')
        require(all(sealed.get('source_snapshot/'+name[5:])==value for name,value in sources.items() if name.startswith('repo/')),'CurveNet source snapshot mismatch')
        selection=bundle.read('selection_manifest.json')
        require(json.loads(selection)==config['selection_manifest'] and config['selection_manifest']['indices']==config['selected_original_ids']
                and config['selection_manifest_identity']==dict(bytes=len(selection),sha256=hashlib.sha256(selection).hexdigest()),'CurveNet dev ID manifest mismatch')
        paths={'batches/'+name+'/batch0000/diagnostics.json' for name in CORRUPTIONS}
        require({name for name in bundle.names if name.startswith('batches/') and name.endswith('/diagnostics.json')}==paths,'CurveNet calibration batch coverage mismatch')
        rows=[];probes=[]
        for name in sorted(paths):
            diagnostic=bundle.json(name);metrics=diagnostic['guidance_metrics'];trajectory=diagnostic['trajectory']
            require(metrics['graph_calls']==metrics['spectral_loss_calls']==0 and metrics['semantic_loss_calls']==2
                    and trajectory['probe_rng_unchanged'] is True and trajectory['probe_buffers_unchanged'] is True
                    and trajectory.get('initial_rng_sha256'),'CurveNet calibration actual calls/RNG/buffers mismatch')
            values=trajectory['curvenet_entropy_calibration']
            require(values['labels_used_for_selection'] is False and values['gradient_calls']==8
                    and values['weighted_calls']==values['kl_calls']==0,'CurveNet calibration executed extra candidate work')
            rows.extend(values['rows']);probes.extend(values['native_probes'])
        report=coefficient_report(rows,config['selected_original_ids']);raw=bundle.read('config.json')
        proof=dict(path=str(Path(path).resolve()),run_id=config['run_id'],config_identity=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
            sealed_manifest_sha256=hashlib.sha256(bundle.read('bundle_manifest.json')).hexdigest(),selected_original_ids=config['selected_original_ids'],
            selection_manifest_identity=config['selection_manifest_identity'],executed_source_files=sources,native_extensions=config['native_extensions'],
            asset_manifest_sha256=config['asset_manifest_sha256'],runtime=config['runtime'],classifier_state_sha256=config['classifier_state_before_sha256'],lion_state=config['lion_state_before'])
        report.update(status='verified_label_free_gpu',calibration_bundles={'scanobjectnn_c':proof},native_probes=probes,
            teacher=dict(checkpoint=CHECKPOINT,checkpoint_identity=config['asset_manifest']['files'][CHECKPOINT],
                         state_sha256=proof['classifier_state_sha256'],preprocessing=PREPROCESS,input_layout='bcn'))
        return validate_report(report)
    finally:bundle.close()


def calibration_worker(directories):
    for directory in directories:
        bundle=runner.source.RunBundle(Path(directory));config=json.loads((bundle.path/'config.json').read_text());validate_run(config)
        config['curvenet_entropy_calibration']=dict(purpose=PURPOSE,labels_used_for_selection=False)
        config['display_name']='Scan CurveNet H-only label-free C-trajectory calibration'
        bundle.write_config(config)
    runner.worker(directories,observer_provider=lambda **values:CurveEntropyCalibrationObserver(**values))


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    if arguments[:1]==['--worker-bundles']:calibration_worker(json.loads(arguments[1]));return
    if arguments[:1]==['summarize']:
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bundle');parser.add_argument('--output',type=Path,required=True)
        args=parser.parse_args(arguments[1:]);report=summarize_bundle(args.bundle);args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
        with args.output.with_suffix('.receipt.json').open('x') as stream:
            stream.write(json.dumps(dict(status='complete',purpose=PURPOSE,report=runner.source.file_identity(args.output)),indent=2)+'\n')
        print(json.dumps(report['coefficients']));return
    require(arguments[:1]==['run'],'Use run <runner options> or summarize <Scan bundle> --output report.json')
    options=['--chamfer-experiment','semantic','--semantic-guidance-head','curvenet','--semantic-kind','marginal_entropy',
             '--semantic-weight','0','--backbones','curvenet',*arguments[1:]]
    configs=runner.configurations(options)
    for config in configs:validate_run(config)
    runner.main(options,worker_script=Path(__file__).resolve())


if __name__=='__main__':main()
