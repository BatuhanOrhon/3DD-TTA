"""Actual CUDA condition and global-chain directional derivative gate."""
import json
from pathlib import Path
import sys


def check(config_path, state_root, output):
    import torch
    config = json.loads(Path(config_path).read_text())
    root = Path(config['asset_root']);sys.path.insert(0, str(root/'source_snapshot'))
    from scripts import eval_shape_conditional as runner
    from shape_latent_conditional import energy, noise_bank
    from shape_latent_refresh import load_proof, save_proof
    from legacy_trajectory import rng_hash
    manifest = runner.shared.source.read_manifest(root, config['asset_manifest_sha256'])
    original = dict(config, method=runner.shared.ORIGINAL_METHOD)
    lion, graph, _, _, _, _ = runner.shared.load_guidance_runtime(original, manifest)
    assert graph is None
    before_models = dict(vae=runner.shared.source.state_hash(lion.vae),
                         priors=runner.shared.source.state_hash(lion.priors))
    assert before_models == config['lion_state_before']
    references = list((Path(state_root)/'firstpass/gaussian').glob('*.npz'))
    assert len(references) == 1
    inputs = json.loads((root/'references/manifest.json').read_text())
    entry = inputs['cases']['gaussian']
    binding = dict(executed_source_sha256=config['executed_source_sha256'],
                   common=entry['common'], input_sha256=entry['proof_binding']['input_sha256'])
    saved = load_proof(references[0], 'cuda', binding)
    h = saved['h_star'][:1].detach()
    scheduler = runner.parent.make_scheduler('cuda')
    bank = noise_bank(h, 71831)
    rng = rng_hash();rows=[];vectors=dict(h_star=h, bank=bank)
    for start_name in ('encoder_z', 'scd_z'):
        start = saved[start_name][:1].detach()
        generator = torch.Generator(device=start.device).manual_seed(71830)
        eps = torch.randn(start.shape, device=start.device, dtype=start.dtype, generator=generator)
        a = scheduler.alphas_cumprod[20].to(start)
        noisy = (a.sqrt()*start+(1-a).sqrt()*eps).detach().requires_grad_()
        def clean_at(u):
            pred = lion.priors[0](x=u,t=u.new_full((len(u),),21),condition_input=None,clip_feat=None)
            return (u-(1-a).sqrt()*pred)/a.sqrt()
        clean = clean_at(noisy)
        _, grad_clean = energy(lion.priors[1],clean,h,scheduler.alphas_cumprod,bank,gradient=True)
        grad_noisy = torch.autograd.grad(clean,noisy,grad_outputs=grad_clean)[0].detach()
        for name, point, grad in [('condition',clean.detach(),grad_clean),('global_chain',noisy.detach(),grad_noisy)]:
            norm = float(grad.norm());assert norm > 0 and torch.isfinite(grad).all()
            direction = grad/grad.norm();trials=[]
            for fraction in (.001,.01):
                delta = fraction*max(float(point.norm()),1.)
                with torch.no_grad():
                    plus = point+delta*direction;minus=point-delta*direction
                    if name == 'global_chain':plus,minus=clean_at(plus),clean_at(minus)
                    vp,_=energy(lion.priors[1],plus,h,scheduler.alphas_cumprod,bank)
                    vm,_=energy(lion.priors[1],minus,h,scheduler.alphas_cumprod,bank)
                    fd=float((vp.mean()-vm.mean())/(2*delta))
                error=abs(fd-norm);relative=error/max(norm,1e-12)
                trials.append(dict(fraction=fraction,delta=delta,finite_difference=fd,
                                   autograd=norm,relative_error=relative,
                                   passed=relative<=.1 or error<=1e-5))
            rows.append(dict(start=start_name,path=name,trials=trials,
                             passed=any(t['passed'] for t in trials)))
            vectors[start_name+'_'+name+'_point']=point
            vectors[start_name+'_'+name+'_gradient']=grad
    after_models=dict(vae=runner.shared.source.state_hash(lion.vae),priors=runner.shared.source.state_hash(lion.priors))
    assert after_models==before_models and rng_hash()==rng
    output=Path(output);output.mkdir()
    save_proof(output/'vectors.npz',vectors,dict(config_identity=runner.shared.source.file_identity(Path(config_path))))
    report=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',
                rows=rows,models_unchanged=True,rng_unchanged=True,source_config=str(config_path),
                peak_memory_bytes=torch.cuda.max_memory_allocated(),torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('CONDITIONAL_NATIVE_GRADIENT_REPORT',json.dumps(report),flush=True)
    if report['status']!='passed':raise RuntimeError('Native gradient gate failed; preserve evidence')
    return report


if __name__=='__main__':
    check(*sys.argv[1:])
