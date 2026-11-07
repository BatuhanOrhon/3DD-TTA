"""Matrix-free physical correction for the proposed research pilot only."""
import math
import torch


def solve(n, edges, edge_weights, node, normals, weights, residual,
          *, kappa=1., strength=1., iterations=100, tolerance=1e-7):
    """Solve a frozen quadratic; return convergence evidence, never hide failure."""
    if not math.isfinite(kappa) or kappa <= 0 or not math.isfinite(strength) or strength < 0:
        raise ValueError('Positive trust and nonnegative graph strength required')
    if any(not torch.isfinite(v).all() for v in (edge_weights,normals,weights,residual)):
        raise ValueError('Nonfinite physical operator')
    if (weights < 0).any() or (edge_weights < 0).any():
        raise ValueError('Negative precision or graph weight')
    if normals.shape != (len(node),3) or residual.shape != weights.shape or len(node)!=len(weights):
        raise ValueError('Observation dimensions differ')
    rhs=normals.new_zeros(n,3)
    rhs.index_add_(0,node,(weights*residual)[:,None]*normals)

    def action(value):
        out=kappa*value
        flux=strength*edge_weights[:,None]*(value[edges[:,0]]-value[edges[:,1]])
        out.index_add_(0,edges[:,0],flux)
        out.index_add_(0,edges[:,1],-flux)
        projected=(normals*value[node]).sum(1)
        out.index_add_(0,node,(weights*projected)[:,None]*normals)
        return out

    degree=weights.new_zeros(n)
    degree.index_add_(0,edges[:,0],edge_weights)
    degree.index_add_(0,edges[:,1],edge_weights)
    blocks=torch.eye(3,device=normals.device,dtype=normals.dtype).expand(n,3,3).clone()
    blocks=blocks*(kappa+strength*degree)[:,None,None]
    blocks.index_add_(0,node,weights[:,None,None]*normals[:,:,None]*normals[:,None,:])
    inverse=torch.linalg.inv(blocks)
    def precondition(value):
        return torch.einsum('nij,nj->ni',inverse,value)

    delta=torch.zeros_like(rhs);r=rhs.clone();z=precondition(r);direction=z.clone()
    rz=(r*z).sum();initial=float((r*r).sum())
    used=0
    if initial:
        for used in range(1,iterations+1):
            hd=action(direction);den=(direction*hd).sum()
            if not torch.isfinite(den) or den<=0:
                raise ValueError('Nonpositive quadratic curvature')
            alpha=rz/den;delta=delta+alpha*direction;r=r-alpha*hd
            if float((r*r).sum())<=tolerance**2*initial:break
            z=precondition(r);following=(r*z).sum()
            direction=z+(following/rz)*direction;rz=following
    actual_residual=rhs-action(delta)
    relative=float(actual_residual.norm())/max(math.sqrt(initial),1e-30)
    change=float(.5*(delta*action(delta)).sum()-(delta*rhs).sum())
    return delta,dict(converged=relative<=tolerance*2,relative_residual=relative,
                      iterations=used,quadratic_change=change)
