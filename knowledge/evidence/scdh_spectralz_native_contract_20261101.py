"""Bounded native distance/GFT witness; no labels, models or experiment changes."""
from pathlib import Path
import hashlib,importlib.util,json,os,shutil,sys,time
import numpy as np
import torch
ROOT=Path('/content/studies/scdh-spectralz-numerical-r3')
FROZEN=ROOT/'prepared/assets/source_snapshot';sys.path.insert(0,str(FROZEN));os.chdir(FROZEN)
from scripts.multibackbone_native_bootstrap import bootstrap
native_proof=bootstrap(ROOT/'prepared/assets');assert native_proof['no_build'] is True
from graph_spectral_v2_offdiag import GraphSpectralDNA
from spectral_guidance_losses import spectral_guidance_loss
assert torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('_contract_tools','/content/deploy_assets_template.py');tools=importlib.util.module_from_spec(spec);spec.loader.exec_module(tools)
output=ROOT/('native_graph_contract_'+str(time.time_ns()));output.mkdir()
plan=json.loads((ROOT/'campaign/plan.json').read_text());manifest=json.loads((ROOT/'prepared/assets/manifest.json').read_text())
for name in ('graph_spectral_v2_offdiag.py','spectral_guidance_losses.py'):tools.verify(FROZEN/name,manifest['files']['source_snapshot/'+name])
cpu=ROOT/'recovery_cpu/current_cpu_preflight_receipt.json';receipt=json.loads(cpu.read_text())
for entry in dict(receipt['native']['bindings'],**receipt['native_extensions']).values():tools.verify(entry['path'],entry)
generator=torch.Generator().manual_seed(20261101)
base=torch.rand(2,32,3,generator=generator)*.12;base[:,-4:]+=torch.tensor([.8,.9,1.1])
results=[]
for label in ('non_tie','duplicate'):
    reference=base.clone()
    if label=='duplicate':reference[:,1]=reference[:,0]
    graph=GraphSpectralDNA(k=10,delta=.1,gamma=.6,use_4d_gft=False,device='cuda')
    points=reference.cuda();captured={}
    hook=graph.knn.register_forward_hook(lambda module,args,result:captured.update(dist=result[0],idx=result[1]))
    coefficients,basis,eigenvalues=graph(points);hook.remove()
    dist,idx=captured['dist'],captured['idx']
    oracle=((reference.double()[:,:,None,:]-reference.double()[:,None,:,:])**2).sum(-1).sqrt()
    actual_dist=dist.cpu().double();actual_idx=idx.cpu().long()
    expected=oracle.topk(10,largest=False,sorted=True)
    torch.testing.assert_close(actual_dist,oracle.gather(2,actual_idx),rtol=1e-4,atol=1e-6)
    torch.testing.assert_close(actual_dist.sort(-1).values,expected.values,rtol=1e-4,atol=1e-6)
    assert all(len(set(row.tolist()))==10 for batch in actual_idx for row in batch)
    assert bool((actual_idx==torch.arange(32)[None,:,None]).any(-1).all())
    if label=='non_tie':assert torch.equal(actual_idx.sort(-1).values,expected.indices.sort(-1).values)
    # Independent CPU graph oracle uses Euclidean distances, RBF, degree pruning and isolates.
    adjacency=torch.zeros(2,32,32,dtype=torch.float64)
    # Validate the realized tie-valid neighbor choice, without assuming a CPU tie order.
    adjacency.scatter_(2,actual_idx,torch.exp(-oracle.gather(2,actual_idx).square()/(2*.1**2)))
    adjacency=torch.maximum(adjacency,adjacency.transpose(1,2));degrees=adjacency.sum(-1)
    valid=degrees>.6*degrees.mean(-1,keepdim=True);adjacency*=valid[:,:,None]*valid[:,None,:]
    nonself=(adjacency>0).sum(-1)-(adjacency.diagonal(dim1=-2,dim2=-1)>0).long()
    laplacian=torch.diag_embed(adjacency.sum(-1))-adjacency+torch.diag_embed((nonself==0).double()*1000)+torch.eye(32)*1e-5
    reconstructed=basis.double().cpu()@torch.diag_embed(eigenvalues.double().cpu())@basis.double().cpu().transpose(1,2)
    torch.testing.assert_close(reconstructed,laplacian,rtol=5e-4,atol=1e-3)
    torch.testing.assert_close(basis@coefficients,points,rtol=1e-4,atol=1e-5)
    prediction=(points+torch.sin(torch.arange(192,device='cuda').reshape(2,32,3))*.01).requires_grad_()
    transformed=basis.transpose(1,2)@prediction
    loss=spectral_guidance_loss(transformed,coefficients,eigenvalues,2.,10,'legacy_l2')
    gradient,=torch.autograd.grad(loss,prediction)
    weights=torch.exp(-2*eigenvalues);weights[:,10:]=0
    analytic=2/(3*32)*(basis@(weights[:,:,None]*(transformed-coefficients)))
    torch.testing.assert_close(gradient,analytic,rtol=1e-4,atol=1e-6)
    np.savez(output/(label+'.npz'),reference=reference.numpy(),distances=dist.cpu().numpy(),indices=idx.cpu().numpy(),basis=basis.cpu().numpy(),eigenvalues=eigenvalues.cpu().numpy(),prediction=prediction.detach().cpu().numpy(),gradient=gradient.cpu().numpy())
    results.append(dict(case=label,batch=2,nodes=32,k=10,euclidean_distance_max_abs_error=float((actual_dist-oracle.gather(2,actual_idx)).abs().max()),laplacian_max_abs_error=float((reconstructed-laplacian).abs().max()),GFT_roundtrip_max_abs_error=float((basis@coefficients-points).abs().max()),gradient_max_abs_error=float((gradient-analytic).abs().max()),graph_statistics=graph.last_statistics))
report=dict(status='accepted',native_bootstrap=native_proof,torch_version=torch.__version__,scope='Two synthetic native FP32 cases; Euclidean kNN distances/top-k/self edge, duplicate-distance-valid selection, pruned Laplacian+isolate penalty+jitter, GFT roundtrip and analytic spectral-coordinate gradient',limitations='No exhaustive tie-order guarantee, no LION Jacobian proof, no accuracy condition or parameter change',source_manifest_sha256=plan['asset_manifest_sha256'],cpu_receipt_identity=tools.identity(cpu),cases=results)
(output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
shutil.copyfile(__file__,output/'audit_source.py')
archive=tools.seal(output,Path(plan['drive_root'])/'recovery-evidence'/(output.name+'.zip'))
(ROOT/'native_graph_contract_archive_receipt.json').write_text(json.dumps(archive,indent=2,sort_keys=True)+'\n')
print(json.dumps(report),flush=True);print('SCDHZ_NATIVE_GRAPH_CONTRACT_ACCEPTED',flush=True)
