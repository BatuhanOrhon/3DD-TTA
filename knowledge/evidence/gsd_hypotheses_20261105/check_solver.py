"""Independent two-node witness: catches edge-weight/sign and zero-data errors."""
import torch
from proximal_solver import solve

normals=torch.tensor([[1.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
node=torch.tensor([0,1]); edges=torch.tensor([[0,1]])
weight=torch.ones(2,dtype=torch.float64); ew=torch.ones(1,dtype=torch.float64)
r=torch.tensor([1.,0.],dtype=torch.float64)
# H_x=[[3,-1],[-1,3]], b=[1,0], so exact x=[3/8,1/8].
d,info=solve(2,edges,ew,node,normals,weight,r,kappa=1.,strength=1.)
torch.testing.assert_close(d,torch.tensor([[.375,0.,0.],[.125,0.,0.]],dtype=torch.float64))
assert info['converged']
z,_=solve(2,edges,ew,node,normals,weight*0,r,kappa=1.,strength=1.)
assert not z.any()
print('Two-node physical solution and zero-observation witnesses passed')

from pilot import graph, normal_observations
points=torch.tensor([(x,y,0.) for x in range(4) for y in range(4)],dtype=torch.float64)
edges,ew,normals,confidence,spacing=graph(points)
assert (edges[:,0]!=edges[:,1]).all()
assert (normals[:,2].abs()>.999999).all()
observed=points+torch.tensor([0.,0.,10.],dtype=torch.float64)
node,ns,ww,rr=normal_observations(points,observed,normals,confidence,spacing)
force=ww[:,None]*rr[:,None]*ns
assert (force[:,2]>0).all()
assert float(force.abs().max())<=1.345/float(spacing)+1e-10
_,ns2,ww2,rr2=normal_observations(points,observed,-normals,confidence,spacing)
torch.testing.assert_close(ww2[:,None]*rr2[:,None]*ns2,force)
print('Planar normal orientation and bounded outlier influence witnesses passed')

# Exact block-Jacobi solves the graph-free, independent normal blocks in one step.
d,info=solve(2,torch.empty((0,2),dtype=torch.long),ew[:0],torch.tensor([0,1]),
             torch.tensor([[1.,0.,0.],[1.,0.,0.]],dtype=torch.float64),
             torch.tensor([1.,100.],dtype=torch.float64),torch.ones(2,dtype=torch.float64),
             kappa=1.,strength=0.)
torch.testing.assert_close(d[:,0],torch.tensor([.5,100/101],dtype=torch.float64))
assert info['iterations']<=1,info
print('Graph-free normal blocks solved in one preconditioned iteration')
