"""Frozen-state GPU mechanism pilot; explicitly not a classification benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from proximal_solver import solve


def identity(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=digest.hexdigest())


def graph(points):
    distances=torch.cdist(points,points).square()
    distances.fill_diagonal_(float('inf'))
    nearest=distances.topk(10,largest=False).indices
    local=points[nearest]-points[:,None]
    covariance=local.transpose(1,2)@local/10
    values,vectors=torch.linalg.eigh(covariance)
    normals=vectors[:,:,0]
    confidence=((values[:,1]-values[:,0])/values[:,2].clamp_min(1e-12)).clamp(0,1)
    source=torch.arange(len(points),device=points.device).repeat_interleave(10)
    dest=nearest.flatten()
    spacing=local.square().sum(2).sqrt().median().clamp_min(1e-6)
    weights=torch.exp(-local.square().sum(2).flatten()/(2*spacing.square()))*.5
    degree=torch.zeros(len(points),device=points.device,dtype=points.dtype)
    degree.index_add_(0,source,weights);degree.index_add_(0,dest,weights)
    weights=weights/degree[degree>0].mean()
    return torch.stack((source,dest),1),weights,normals,confidence,spacing


def normal_observations(points,observed,normals,confidence,sigma):
    node=torch.cdist(observed,points).argmin(1)
    ns=normals[node]
    residual=((observed-points[node])*ns).sum(1)
    # Local spacing is a robust scale, not measured corruption variance.
    weights=confidence[node]/sigma.square()*torch.minimum(torch.ones_like(residual),1.345*sigma/residual.abs().clamp_min(1e-12))
    return node,ns,weights,residual


def clean_cd(points,clean):
    distance=torch.cdist(points,clean).square()
    return float(distance.min(1).values.mean()+distance.min(0).values.mean())


def capped(direction,points,rho=.001):
    return direction*(rho*points.norm()/direction.norm().clamp_min(1e-12)).clamp(max=1.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',choices=('anchors','transfer'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--ids',nargs=2,type=int,default=[44,91])
    parser.add_argument('--seed',type=int,choices=(0,1,2),default=0)
    parser.add_argument('--corruptions',nargs='+',default=['gaussian','background','shear','lidar'])
    parser.add_argument('--timestep',type=int,choices=(10,70,340))
    args=parser.parse_args()
    assert len(set(args.ids))==2 and all(0<=v<2468 for v in args.ids)
    allowed={'uniform','gaussian','background','impulse','upsampling','distortion_rbf','distortion_rbf_inv','density','density_inc','shear','rotation','cutout','distortion','occlusion','lidar'}
    assert set(args.corruptions)<=allowed and len(set(args.corruptions))==len(args.corruptions)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    metadata=dict(group=args.group,ids=args.ids,seed=args.seed,batch_size=2,
                  corruptions=args.corruptions,severity=5,timestep_override=args.timestep,
                  scope='first noisy-to-clean prediction, not full SCD trajectory, accuracy or throughput',
                  setup=json.loads(Path('/content/cli_setup_verification.json').read_text()),
                  source={f.name:identity(f) for f in Path(__file__).parent.glob('*.py')})
    assert metadata['setup']['status']=='passed'
    checkpoint=Path('/content/3DD-TTA/lion_ckpts/epoch_10999_iters_2100999.pt')
    configuration=Path('/content/3DD-TTA/lion_ckpts/unconditional_all55_cfg.yml')
    metadata['lion_checkpoint']=identity(checkpoint)
    metadata['lion_configuration']=identity(configuration)
    assert metadata['lion_checkpoint']['sha256']=='807f6732ad087a1ffdaeeaa456e32b4130c9a8a1708446cabc0059654a0a86c2'
    assert metadata['lion_configuration']['sha256']=='9d4a826f35fe9b3053210105eb4857811c8bfd53cd8c6a998f8ea7953e975816'
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,out/f.name)
    sys.path.insert(0,'/content/3DD-TTA')
    from default_config import cfg
    from models.lion import LION
    from diffusers import DDIMScheduler
    import utilities_3dd_tta as ops
    from graph_spectral import SpectralConfig,build_smooth_spectral_target
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    options=cfg.clone();options.merge_from_file('/content/3DD-TTA/lion_ckpts/unconditional_all55_cfg.yml')
    lion=LION(options);lion.load_model('/content/3DD-TTA/lion_ckpts/epoch_10999_iters_2100999.pt')
    lion.vae.eval().requires_grad_(False);lion.priors.eval().requires_grad_(False)
    root=Path('/content/3DD-TTA/data/modelnet40_c')
    original=np.load(root/'data_original.npy',mmap_mode='r')
    records=[];started=time.monotonic()
    try:
        for corruption in metadata['corruptions']:
            if time.monotonic()-started>1200:raise RuntimeError('20-minute phase cap')
            torch.manual_seed(args.seed);np.random.seed(args.seed)
            path=root/('data_'+corruption+'_5.npy')
            actual_identity=identity(path)
            expected=metadata['setup']['assets'][str(path.relative_to('/content/3DD-TTA'))]
            assert all(actual_identity[k]==expected[k] for k in ('bytes','sha256'))
            (out/(corruption+'_data_identity.json')).write_text(json.dumps(actual_identity,indent=2)+'\n')
            raw=np.array(np.load(path,mmap_mode='r')[metadata['ids']],dtype=np.float32,copy=True)
            normalized,center,scale=ops.normalize(torch.from_numpy(raw))
            x=ops.rotate_pointcloud(torch.from_numpy(ops.upsample_all(normalized.numpy(),2048)).float().cuda()*3.3885)
            clean=ops.rotate_pointcloud(((torch.from_numpy(np.array(original[metadata['ids']],dtype=np.float32,copy=True))-center)/scale).cuda()*3.3885)
            scheduler=DDIMScheduler(beta_start=.0001,beta_end=.02,beta_schedule='linear',num_train_timesteps=1000,clip_sample=False,prediction_type='epsilon',set_alpha_to_one=True)
            scheduler.set_timesteps(100,device='cuda')
            t=scheduler.timesteps[-(35 if corruption=='background' else 8)] if args.timestep is None else torch.tensor(args.timestep,device='cuda')
            with torch.no_grad():
                encoded=lion.vae.encode(x)
                z=encoded[2][0][0];h=encoded[2][1][0]
                style=z.unsqueeze(2).unsqueeze(3)
                h=h.unsqueeze(2).unsqueeze(3)
                a=scheduler.alphas_cumprod[t].sqrt();sigma=(1-a*a).sqrt()
                noisy=a*h+sigma*torch.randn_like(h)
                condition=lion.vae.global2style(style)
                def prior():return lion.priors[1](x=noisy,t=torch.ones(2,device='cuda')*(t+1),condition_input=condition,clip_feat=None)
                epsilon=prior();repeated=prior()
                state=scheduler.step(epsilon,t,noisy)
                latent=state.pred_original_sample
                def decode(v):return lion.vae.decoder(None,beta=None,context=v.squeeze(3).squeeze(2),style=z)
                y=decode(latent);repeat_y=decode(latent);anchor=decode(h)
            np.savez_compressed(out/(corruption+'_states.npz'),x=x.cpu().numpy(),clean=clean.cpu().numpy(),y=y.cpu().numpy(),anchor=anchor.cpu().numpy(),latent=latent.cpu().numpy(),z=z.cpu().numpy(),noisy=noisy.cpu().numpy(),epsilon=epsilon.cpu().numpy(),normalization_center=center.numpy(),normalization_scale=scale.numpy(),repeated_epsilon=repeated.cpu().numpy(),repeated_y=repeat_y.cpu().numpy())
            filters=build_smooth_spectral_target(x,SpectralConfig(),profile='smooth',beta=2.) if args.group=='anchors' else None
            if filters is not None:
                np.savez_compressed(out/(corruption+'_actual_filters.npz'),**{'filter'+str(i):v.cpu().numpy() for i,v in enumerate(filters.filters)})
            for index,object_id in enumerate(metadata['ids']):
                yy,xx,cc=y[index],x[index],clean[index]
                edges,ew,normals,confidence,spacing=graph(yy.double())
                node,ns,ww,rr=normal_observations(yy.double(),xx.double(),normals,confidence,spacing)
                np.savez_compressed(out/(corruption+'_'+str(object_id)+'_operator.npz'),edges=edges.cpu().numpy(),edge_weights=ew.cpu().numpy(),normals=normals.cpu().numpy(),confidence=confidence.cpu().numpy(),node=node.cpu().numpy(),weights=ww.cpu().numpy(),residual=rr.cpu().numpy(),spacing=spacing.cpu().numpy())
                common=dict(corruption=corruption,object_id=object_id,baseline_clean_cd=clean_cd(yy,cc),
                            prior_repeat_rms=float((epsilon-repeated).square().mean().sqrt()),
                            decoder_repeat_rms=float((y-repeat_y).square().mean().sqrt()))
                directions={}
                if args.group=='anchors':
                    f=filters.filters[index]
                    directions['H1_raw_anchor']=-(f@(yy-xx))
                    directions['H1_reconstructed_anchor']=-(f@(yy-anchor[index]))
                    normal_rhs=yy.double().new_zeros(2048,3)
                    normal_rhs.index_add_(0,node,(ww*rr)[:,None]*ns)
                    pn,pns,pw,pr=normal_observations(yy.double(),xx.double().flip(0),normals,confidence,spacing)
                    permuted=torch.zeros_like(normal_rhs)
                    permuted.index_add_(0,pn,(pw*pr)[:,None]*pns)
                    records.append(dict(common,method='H2_observation_permutation',
                        relative_rhs_error=float((permuted-normal_rhs).norm()/normal_rhs.norm().clamp_min(1e-12))))
                    directions['H2_robust_surface']=normal_rhs.float()
                else:
                    for strength in (0.,1.):
                        delta,info=solve(2048,edges,ew,node,ns,ww,rr,kappa=1.,strength=strength,tolerance=1e-6)
                        records.append(dict(common,method='H3_solver',strength=strength,**info))
                        if not info['converged'] or info['quadratic_change']>1e-10:
                            raise RuntimeError('Sparse solve gate failed')
                        directions['H3_lambda'+str(strength)]=delta.float()
                    for rho in (.001,.01):
                        delta=capped(directions['H3_lambda1.0'],yy,rho)
                        lift=torch.zeros_like(latent);lift.reshape(2,2048,4)[index,:,:3]=delta
                        with torch.no_grad():
                            changed=decode(latent+lift)[index]
                            eps_new=(noisy-a*(latent+lift))/sigma
                            actual=scheduler.step(eps_new,t,noisy).prev_sample-state.prev_sample
                            s=int(t)-10
                            alpha_s=scheduler.alphas_cumprod[s] if s>=0 else scheduler.final_alpha_cumprod
                            coefficient=alpha_s.sqrt()-(1-alpha_s).sqrt()*a/sigma
                            expected=coefficient*lift
                        np.savez_compressed(out/(corruption+'_'+str(object_id)+'_lift_'+str(rho)+'.npz'),delta=delta.cpu().numpy(),changed=changed.cpu().numpy(),actual_scheduler_difference=actual.cpu().numpy(),expected_scheduler_difference=expected.cpu().numpy(),coefficient=coefficient.cpu().numpy())
                        records.append(dict(common,method='H4_transfer',rho=rho,
                            finite_lift_relative_error=float((changed-yy-delta).norm()/delta.norm().clamp_min(1e-12)),
                            finite_lift_cosine=float(torch.nn.functional.cosine_similarity((changed-yy).flatten(),delta.flatten(),dim=0)),
                            scheduler_max_error=float((actual-expected).abs().max()),
                            requested_delta_norm=float(delta.norm())))
                for name,direction in directions.items():
                    delta=capped(direction,yy)
                    after=clean_cd(yy+delta,cc)
                    np.savez_compressed(out/(corruption+'_'+str(object_id)+'_'+name+'_delta.npz'),delta=delta.cpu().numpy())
                    records.append(dict(common,method=name,delta_norm=float(delta.norm()),clean_cd_after=after,
                                        clean_cd_change=after-common['baseline_clean_cd']))
            torch.cuda.synchronize()
            (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
            print('CASE_COMPLETE',args.group,corruption,len(records),time.monotonic()-started,flush=True)
        (out/'complete.json').write_text(json.dumps(dict(status='complete',seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),classification_accuracy_measured=False),indent=2)+'\n')
    except Exception as error:
        (out/'records.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error),seconds=time.monotonic()-started))+'\n')
        raise


if __name__=='__main__':main()
