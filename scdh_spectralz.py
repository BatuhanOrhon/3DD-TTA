"""Opt-in fixed-basis spectral style gradients; local SCD remains separate."""
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from legacy_trajectory import rng_hash, tensor_hash
from spectral_guidance_losses import spectral_guidance_loss


def sample_norm(value):
    return value.detach().reshape(len(value), -1).double().norm(dim=1)


class SpectralStyleRoute:
    def __init__(self, graph, modes, kappa, *, cache_path=None, audit_path=None):
        if (type(modes) is not int or not 1 <= modes <= 2048
                or isinstance(kappa, bool) or not isinstance(kappa, (float, int))
                or not math.isfinite(kappa) or kappa < 0):
            raise ValueError('Require positive M<=2048 and finite nonnegative spectral-z step')
        self.graph, self.modes, self.style_step = graph, modes, float(kappa)
        self.cache_path = None if cache_path is None else Path(cache_path)
        self.reference = self.basis = self.eigenvalues = self.target = None
        self.records, self.graph_record = [], {}
        self.audit_path = None if audit_path is None else Path(audit_path)
        self.audit_record = None

    def initialize(self, reference):
        if (self.reference is not None or reference.ndim != 3 or reference.shape[-1] != 3
                or reference.dtype not in (torch.float32, torch.float64)
                or not torch.isfinite(reference).all()):
            raise ValueError('Require one initialization with finite BxNx3 reference')
        before, started = rng_hash(), time.perf_counter()
        graph_file = Path(inspect.getfile(type(self.graph)))
        binding = dict(reference_sha256=tensor_hash(reference),
                       graph_source_sha256=hashlib.sha256(graph_file.read_bytes()).hexdigest(),
                       route_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       graph_parameters={key: getattr(self.graph, key, None)
                                         for key in ('k', 'delta', 'gamma', 'use_4d_gft')})
        cached = self.cache_path is not None and self.cache_path.exists()
        if self.cache_path is not None and (self.cache_path.is_symlink()
                or any(parent.is_symlink() for parent in self.cache_path.parents)):
            raise ValueError('Spectral cache symlinks are forbidden')
        if cached:
            with np.load(self.cache_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive['metadata'].item()))
                if metadata.get('binding') != binding:
                    raise ValueError('Spectral graph cache identity mismatch')
                basis, eigenvalues = (torch.from_numpy(archive[key].copy()).to(reference.device)
                                      for key in ('basis', 'eigenvalues'))
                for key, value in (('basis', basis), ('eigenvalues', eigenvalues)):
                    if tensor_hash(value) != metadata['tensor_hashes'].get(key):
                        raise ValueError('Spectral graph cache tensor hash mismatch: ' + key)
        else:
            with torch.no_grad():
                _, basis, eigenvalues = self.graph(reference.detach())
            metadata = dict(binding=binding, pruning=getattr(self.graph,'last_statistics',None), tensor_hashes=dict(basis=tensor_hash(basis),
                                                               eigenvalues=tensor_hash(eigenvalues)))
        batch, count, _ = reference.shape
        if (basis.shape != (batch, count, count) or eigenvalues.shape != (batch, count)
                or basis.dtype != reference.dtype or eigenvalues.dtype != reference.dtype
                or not torch.isfinite(basis).all() or not torch.isfinite(eigenvalues).all()
                or not bool((eigenvalues[:, 1:] >= eigenvalues[:, :-1]).all())):
            raise ValueError('Invalid finite ordered spectral eigensystem')
        if not cached and self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Runtime-only full basis; archived batch diagnostics retain its identity.
            with self.cache_path.open('xb') as stream:
                np.savez(stream, basis=basis.detach().cpu().numpy(),
                         eigenvalues=eigenvalues.detach().cpu().numpy(),
                         metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
        self.reference, self.basis, self.eigenvalues = reference.detach().clone(), basis.detach(), eigenvalues.detach()
        self.target = (self.basis.transpose(1, 2) @ self.reference).detach()
        weights = torch.exp(-2 * self.eigenvalues)
        weights[:, self.modes:] = 0
        if not torch.isfinite(weights).all():
            raise FloatingPointError('Nonfinite spectral filter weights')
        boundary = min(self.modes, count)
        self.graph_record = dict(binding=binding, basis_sha256=tensor_hash(self.basis),
            pruning=metadata.get('pruning'), cache_file_sha256=hashlib.sha256(self.cache_path.read_bytes()).hexdigest() if self.cache_path is not None else None,
            eigenvalues_sha256=tensor_hash(self.eigenvalues), reference_spectrum_sha256=tensor_hash(self.target),
            profile='offdiag_static_raw_lambda_truncated_l2', beta=2., M_max=self.modes,
            reduction='batch sum / (3*N)', cache_mode='runtime_basis_replay' if cached else 'fresh_eigh',
            eigenvalues=self.eigenvalues.cpu().tolist(), weights_sha256=tensor_hash(weights),
            effective_spectral_mass=weights.sum(1).cpu().tolist(), nonzero_weights=weights.count_nonzero(1).cpu().tolist(),
            isolation_penalty_modes=(self.eigenvalues > 999).sum(1).cpu().tolist(),
            cutoff_gap=(self.eigenvalues[:, boundary] - self.eigenvalues[:, boundary-1]).cpu().tolist() if boundary < count else None,
            build_or_load_seconds=time.perf_counter() - started,
            archive_contract='full eigenvalues and basis/weight hash; full runtime basis is not a portable reconstruction guarantee')
        if rng_hash() != before:
            raise RuntimeError('Spectral graph initialization advanced the main RNG')

    def loss(self, clean):
        if self.reference is None or clean.shape != self.reference.shape:
            raise ValueError('Expected initialized, matching finite clean latent XYZ')
        if not torch.isfinite(clean).all():raise FloatingPointError('Nonfinite predicted-clean spectral input')
        prediction = self.basis.transpose(1, 2) @ clean
        value = spectral_guidance_loss(prediction, self.target, self.eigenvalues, 2., self.modes)
        if not torch.isfinite(value):
            raise FloatingPointError('Nonfinite spectral loss')
        return value

    def gradients(self, clean, scd_loss, local, style, step, timestep):
        if not torch.isfinite(scd_loss):
            raise FloatingPointError('Nonfinite SCD loss')
        local_scd, style_scd = torch.autograd.grad(scd_loss, (local, style), retain_graph=True)
        spectral_loss = self.loss(clean)
        style_spectral, = torch.autograd.grad(spectral_loss, style)
        if not all(bool(torch.isfinite(value).all()) for value in (local_scd, style_scd, style_spectral)):
            raise FloatingPointError('Nonfinite routed gradient')
        c, s = (value.detach().reshape(len(value), -1).double() for value in (style_scd, style_spectral))
        cn, sn = c.norm(dim=1), s.norm(dim=1)
        denominator = cn * sn
        cosines = [float(dot / norm) if norm > 0 else None for dot, norm in zip((c*s).sum(1), denominator)]
        self.records.append(dict(loop_count=step+1, timestep=int(timestep), scd_loss=float(scd_loss.detach()),
            spectral_loss=float(spectral_loss.detach()), local_scd_gradient_norm=sample_norm(local_scd).cpu().tolist(),
            style_scd_gradient_norm=cn.cpu().tolist(), style_spectral_gradient_norm=sn.cpu().tolist(),
            style_scd_spectral_cosine=cosines, kappa_z_spectral_effective=self.style_step,
            applied_local_spectral_norm=[0.] * len(local), applied_style_scd_norm=[0.] * len(style),
            applied_style_spectral_norm=(self.style_step * sn).cpu().tolist()))
        return local_scd, style_spectral

    def diagnostics(self):
        return dict(graph=self.graph_record, steps=self.records,
                    native_finite_difference=self.audit_record,
                    global_diffusion=False, h_route='SCD only', z_route='spectral only',
                    original_z_decode=True, final_style_update_consumed_by_decode=False)

    def audit(self, clean, local, style, local_gradient, style_gradient, clean_function):
        if self.audit_path is None or self.audit_record is not None:
            return
        if len(style)<2:
            raise ValueError('Native directional gate requires two physical examples')
        checks, vectors = [], {}
        rng_before=rng_hash()
        inputs_before=dict(local=tensor_hash(local),style=tensor_hash(style))
        with torch.no_grad():
            repeat=clean_function(style.detach())
            repeat_delta=sample_norm(repeat-clean)
            for index in range(2):
                direction=torch.zeros_like(style)
                pattern=torch.arange(style[index].numel(),device=style.device).reshape_as(style[index])
                direction[index]=torch.where(pattern%2==0,1.,-1.)
                direction[index]/=direction[index].norm()
                norm=style[index].norm()
                gradient_norm=style_gradient[index].norm()
                if not torch.isfinite(norm) or norm<=0 or not torch.isfinite(gradient_norm) or gradient_norm<=0:
                    raise RuntimeError('Native finite-difference requires nonzero finite style and gradient')
                exact=float((style_gradient*direction).sum())
                def loss_at(z):
                    physical=clean_function(z)
                    spectrum=self.basis[index:index+1].transpose(1,2)@physical[index:index+1]
                    return spectral_guidance_loss(spectrum,self.target[index:index+1],self.eigenvalues[index:index+1],2.,self.modes)
                for scale in (.001,.01):
                    epsilon=float(norm)*scale
                    plus,minus=loss_at(style.detach()+epsilon*direction),loss_at(style.detach()-epsilon*direction)
                    estimate=float((plus-minus)/(2*epsilon))
                    absolute=abs(estimate-exact)
                    relative=absolute/max(abs(exact),abs(estimate)) if max(abs(exact),abs(estimate))>0 else 0.
                    checks.append(dict(example=index,scale=scale,epsilon=epsilon,autograd=exact,
                        loss_plus=float(plus),loss_minus=float(minus),finite_difference=estimate,absolute_error=absolute,relative_error=relative,
                        passed=math.isfinite(estimate) and (relative<=.1 or absolute<=1e-5)))
                vectors['direction_%d'%index]=direction[index].cpu().numpy()
        self.audit_record=dict(checks=checks,repeat_clean_delta_norm=repeat_delta.cpu().tolist(),
            relative_tolerance=.1,absolute_tolerance=1e-5,path=str(self.audit_path),
            rng_before_sha256=rng_before,rng_after_sha256=rng_hash(),
            inputs_before=inputs_before,inputs_after=dict(local=tensor_hash(local),style=tensor_hash(style)),
            passed=all(c['passed'] for c in checks))
        self.audit_record['passed'] &= (rng_before==self.audit_record['rng_after_sha256'] and inputs_before==self.audit_record['inputs_after'])
        payload={name:value.detach()[:2].cpu().numpy() for name,value in
            dict(basis=self.basis,eigenvalues=self.eigenvalues,reference=self.reference,clean=clean,
                 repeat_clean=repeat,local=local,style=style,local_gradient=local_gradient,style_gradient=style_gradient).items()}
        payload.update(vectors)
        self.audit_record['physical_hashes']={name:hashlib.sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()
                                              for name,value in payload.items()}
        self.audit_path.parent.mkdir(parents=True,exist_ok=True)
        with self.audit_path.open('xb') as stream:
            np.savez_compressed(stream,**payload,
                metadata=np.asarray(json.dumps(self.audit_record,sort_keys=True)))
        self.audit_record['sha256']=hashlib.sha256(self.audit_path.read_bytes()).hexdigest()
        if not self.audit_record['passed']:
            raise RuntimeError('Native finite-difference gate failed; physical evidence preserved')


class SpectralCalibrationRoute(SpectralStyleRoute):
    """Observe spectral scales on the unchanged C_base trajectory, without labels."""
    def __init__(self, graph, checkpoints, *, modes=(200, 400, 800, 1200, 1800), cache_path=None):
        super().__init__(graph, max(modes), .003, cache_path=cache_path)
        self.checkpoints, self.calibration_modes = tuple(checkpoints), tuple(modes)
        self.calibration = []

    def gradients(self, clean, scd_loss, local, style, step, timestep):
        observe = step+1 in self.checkpoints
        local_scd, style_scd = torch.autograd.grad(scd_loss, (local, style), retain_graph=observe)
        if not all(bool(torch.isfinite(v).all()) for v in (local_scd, style_scd)):
            raise FloatingPointError('Nonfinite calibration SCD gradient')
        if observe:
            scd_norm = sample_norm(style_scd)
            for modes in self.calibration_modes:
                self.modes = modes
                gradient, = torch.autograd.grad(self.loss(clean), style, retain_graph=True)
                spectral_norm = sample_norm(gradient)
                ratios = [.003 * float(c) / float(s) if math.isfinite(float(c)) and math.isfinite(float(s)) and s>0 else None
                          for c,s in zip(scd_norm, spectral_norm)]
                self.calibration.append(dict(loop_count=step+1, timestep=int(timestep), M_max=modes,
                    weights_sha256=tensor_hash(torch.exp(-2*self.eigenvalues)*
                        (torch.arange(self.eigenvalues.shape[1],device=self.eigenvalues.device)[None,:]<modes)),
                    style_scd_gradient_norm=scd_norm.cpu().tolist(),
                    style_spectral_gradient_norm=[float(v) if math.isfinite(float(v)) else None for v in spectral_norm],
                    excluded_nonfinite=[not math.isfinite(float(v)) for v in spectral_norm],ratios=ratios))
        return local_scd, style_scd

    def diagnostics(self):
        return dict(graph=self.graph_record, calibration=self.calibration, global_diffusion=False,
                    h_route='SCD only', z_route='SCD only (calibration observation)',
                    original_z_decode=True, final_style_update_consumed_by_decode=False)
