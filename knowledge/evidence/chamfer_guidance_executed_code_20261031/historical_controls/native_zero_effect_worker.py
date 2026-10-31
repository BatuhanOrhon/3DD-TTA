"""Exact pointwise native zero-weight gradient diagnostic on the frozen sampler."""
import json
import os
from pathlib import Path
import sys
import traceback


def worker(directories, spec):
    import torch
    from legacy_trajectory import rng_hash
    from semantic_trajectory import SemanticTrajectoryObserver
    from scripts import eval_multibackbone_guidance as runner
    output = Path(spec['sidecar_root']); output.mkdir(parents=True,exist_ok=False)
    rows = []
    def observer_provider(**values):
        corruption = values.pop('corruption'); values.pop('original_ids')
        observer = SemanticTrajectoryObserver(**values)
        local_rows = []
        original_diagnostics = observer.diagnostics
        def probe(**arguments):
            loss = arguments['semantic_loss']
            if loss is None: return
            local,style = arguments['noisy_local'],arguments['style']
            before_rng = rng_hash()
            before = [None if tensor.grad is None else tensor.grad.detach().clone() for tensor in (local,style)]
            gradients = torch.autograd.grad(0.*loss,(local,style),retain_graph=True)
            record = dict(corruption=corruption,loop_count=arguments['loop_count'],kind=observer.kind,
                          batch_count=len(local),raw_semantic_loss=float(loss.detach()))
            for name,gradient in zip(('local','style'),gradients):
                finite=bool(torch.isfinite(gradient).all()); nonzero=int(torch.count_nonzero(gradient))
                record[name]=dict(finite=finite,nonzero_count=nonzero,max_abs=float(gradient.abs().max()),norm=float(gradient.norm()))
                if not finite or nonzero != 0: raise RuntimeError('Native zero semantic gradient is not exactly zero: '+name)
            unchanged = all((tensor.grad is None if saved is None else tensor.grad is not None and torch.equal(tensor.grad,saved))
                            for tensor,saved in zip((local,style),before))
            record.update(leaf_grad_fields_unchanged=unchanged,rng_unchanged=rng_hash()==before_rng)
            if not unchanged or not record['rng_unchanged']: raise RuntimeError('Zero diagnostic changed .grad or RNG')
            rows.append(record);local_rows.append(record)
        def diagnostics():
            result=original_diagnostics();result['native_zero_effect']=local_rows;return result
        observer.guidance_components=probe;observer.diagnostics=diagnostics
        return observer
    try:
        runner.worker(directories,observer_provider=observer_provider)
        if len(rows)!=4 or {(r['corruption'],r['loop_count']) for r in rows}!={('gaussian',5),('gaussian',8),('background',15),('background',23)}:
            raise RuntimeError('Native zero-effect checkpoint coverage incomplete')
    except BaseException:
        (output/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        with (output/'zero_rows.json').open('x') as stream:json.dump(rows,stream,indent=2,allow_nan=False)


def main():
    spec=json.loads(Path(os.environ['CHAMFER_NATIVE_DIAGNOSTIC_SPEC']).read_text())
    sys.path.insert(0,spec['source_root'])
    if sys.argv[1:2]==['--worker-bundles']:
        worker(json.loads(sys.argv[2]),spec);return
    from scripts import eval_multibackbone_guidance as runner
    runner.main(spec['effective_runner_arguments'],worker_script=Path(__file__).resolve())


if __name__=='__main__':main()
