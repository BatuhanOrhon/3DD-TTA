import os, sys, json, hashlib, subprocess, importlib
from pathlib import Path
os.chdir('/content/3DD-TTA')
sys.path.insert(0,os.getcwd())
assert sys.version_info[:2] == (3, 8), sys.version
import torch, numpy as np
assert torch.cuda.is_available()
assert 'A100' in torch.cuda.get_device_name(0)
a=torch.arange(32,device='cuda',dtype=torch.float32)
assert float(a.sum().cpu())==496
modules=['knn_cuda','emd_cuda','chamfer','pointnet2_ops._ext','clip','open3d','h5py','diffusers','timm','models.pvcnn2','utils.eval_helper']
imports={name:str(importlib.import_module(name).__file__) for name in modules}
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL2
from extensions.emd import emd
x=torch.rand(1,32,3,device='cuda').contiguous()
i=pointnet2_utils.furthest_point_sample(x,8)
assert tuple(i.shape)==(1,8) and bool(((i>=0)&(i<32)).all())
d,idx=KNN(k=4,transpose_mode=True)(x,x)
assert bool(torch.isfinite(d).all()) and idx.numel()==128
cd=ChamferDistanceL2()(x,x)
assert bool(torch.isfinite(cd).all()) and abs(float(cd))<1e-5
cost=emd()(x,x)
assert bool(torch.isfinite(cost).all())
def identity(p):
 p=Path(p);assert p.is_file() and p.stat().st_size>0,str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
assets={s:identity(s) for s in ['env.yaml','requirements.txt','scripts/build_cuda_extensions_colab.sh','/content/KNN_CUDA-0.2-py3-none-any.whl','pointnet_ckpts/modelnet_jt.pth','lion_ckpts/epoch_10999_iters_2100999.pt']}
root=Path('data/modelnet40_c')
assert root.is_dir()
clean=np.load(str(root/'data_original.npy'),mmap_mode='r')
assert clean.ndim==3 and clean.shape[0]==2468 and clean.shape[2]==3,clean.shape
assets[str(root/'data_original.npy')]=identity(root/'data_original.npy')
assets['data/modelnet40_c.zip']=identity('data/modelnet40_c.zip')
arrays=sorted(root.glob('data_*_5.npy'))
assert len(arrays)>=15,len(arrays)
assert (root/'label.npy').is_file()
assets[str(root/'label.npy')]=identity(root/'label.npy')
for f in arrays:
 assets[str(f)]=identity(f)
 z=np.load(str(f),mmap_mode='r');assert z.ndim==3 and z.shape[0]==2468 and z.shape[2]==3,(str(f),z.shape)
assert Path('/content/drive/MyDrive/thesis/dataset').is_dir()
info={'status':'passed','python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,'meminfo':Path('/proc/meminfo').read_text().splitlines()[:3],'remote_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'remote_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'remote_status':subprocess.check_output(['git','status','--short'],text=True),'imports':imports,'native_checks':['CUDA tensor sum','FPS','KNN','Chamfer identical input','EMD finite output'],'assets':assets,'modelnet_corruption_arrays':len(arrays)}
Path('/content/cli_setup_verification.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2));print('SETUP_ENVIRONMENT_VERIFIED')
