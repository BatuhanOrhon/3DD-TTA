"""Independent SciPy sparse-matrix oracle on the actual GPU technical states."""
import io,json,zipfile
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
root=Path('result/preparation/gsd_hypotheses_20261105')
records=[]
with zipfile.ZipFile(root/'transfer_v3_technical.zip') as archive:
    rows=json.loads(archive.read('science/records.json'))
    def load(name):
        with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as value:return dict(value)
    for corruption in ('gaussian','background','shear','lidar'):
        state=load('science/'+corruption+'_states.npz')
        for index,object_id in enumerate((44,91)):
            op=load('science/'+corruption+'_'+str(object_id)+'_operator.npz')
            n=2048;edge=op['edges'];m=len(op['node']);e=len(edge)
            incidence=sparse.coo_matrix((np.tile([1.,-1.],e),(np.repeat(np.arange(e),2),edge.flatten())),shape=(e,n)).tocsr()
            laplacian=incidence.T@sparse.diags(op['edge_weights'])@incidence
            normal=op['normals'][op['node']]
            observation=sparse.coo_matrix((normal.flatten(),(np.repeat(np.arange(m),3),(3*op['node'][:,None]+np.arange(3)).flatten())),shape=(m,3*n)).tocsr()
            gram=observation.T@sparse.diags(op['weights'])@observation
            rhs=observation.T@(op['weights']*op['residual'])
            for strength in (0.,1.):
                matrix=sparse.eye(3*n,format='csr')+strength*sparse.kron(laplacian,sparse.eye(3),format='csr')+gram
                solution,info=cg(matrix,rhs,rtol=1e-10,atol=0.,maxiter=3000)
                assert info==0,(corruption,object_id,strength,info)
                residual=np.linalg.norm(matrix@solution-rhs)/max(np.linalg.norm(rhs),1e-30)
                delta=solution.reshape(n,3).astype(np.float32)
                factor=min(1.,.001*np.linalg.norm(state['y'][index])/max(np.linalg.norm(delta),1e-12))
                expected=delta*factor
                actual=load('science/'+corruption+'_'+str(object_id)+'_H3_lambda'+str(strength)+'_delta.npz')['delta']
                error=float(np.max(abs(actual-expected)))
                optimal=-.5*float(rhs@solution)
                gpu=next(v for v in rows if v['corruption']==corruption and v['object_id']==object_id and v['method']=='H3_solver' and v['strength']==strength)
                value_error=abs(optimal-gpu['quadratic_change'])/max(abs(optimal),1e-12)
                assert error<1e-6 and value_error<1e-6,(error,value_error)
                records.append(dict(corruption=corruption,object_id=object_id,strength=strength,cpu_residual=residual,capped_delta_max_error=error,relative_optimal_value_error=value_error))
(root/'independent_scipy_sparse.json').write_text(json.dumps({'status':'passed','rows':records},indent=2)+'\n')
print('Independent SciPy sparse cases',len(records),'maxdelta',max(v['capped_delta_max_error'] for v in records),'maxvalue',max(v['relative_optimal_value_error'] for v in records))
