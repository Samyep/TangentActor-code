"""Portable orchestration of the extracted, unchanged numerical kernels."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .ambient import TerminalAwareActor
from .benchmarks import HJBLogBenchmark, AllenCahnBenchmark, OUGMMBenchmark, MLPFailureBenchmark
from .cosine_hjb import CosineHJBBenchmark
from .drivers import JointConcaveDriver, JointConvexOriginalDriver, SignTransformedDriver
from .sine_drivers import SineQuadraticDriver
from . import feynman_kac as fk
from . import joint_models, joint_training, joint_mcv, joint_fk, joint_evaluation
from .low_rank import LearnedLowRankTerminalAwareActor, train_low_rank_actor
from .utils import seed_everything

PDES = ('hjb_log','ou_gmm','allen_cahn','joint_concave','joint_convex','nonconvex_coupled','mlp_failure','b6_fullrank','dense_fourier','cosine_hjb')


def data_path(name: str) -> Path:
    """Resolve a locally supplied benchmark input."""
    root = os.environ.get("TANGENT_ACTOR_DATA_DIR")
    if not root:
        raise FileNotFoundError(f"Set TANGENT_ACTOR_DATA_DIR to the directory containing {name}")
    path = Path(root).expanduser() / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path



def make_problem(pde: str, dimension: int):
    d=int(dimension)
    if pde=='hjb_log': return HJBLogBenchmark(d)
    if pde == 'cosine_hjb': return CosineHJBBenchmark(d)
    if pde=='allen_cahn': return AllenCahnBenchmark(d)
    if pde=='ou_gmm': return OUGMMBenchmark.from_npz(data_path(f'ou_gmm_d{d}.npz'))
    if pde=='mlp_failure': return MLPFailureBenchmark(d)
    if pde=='joint_concave': return JointConcaveDriver(d=d,k=2,T=.5)
    if pde=='joint_convex': return SignTransformedDriver(JointConvexOriginalDriver(d=d,k=2,T=.5))
    if pde=='nonconvex_coupled': return SineQuadraticDriver(d=d,T=.5,lam=.2,eta=1.)
    if pde=='dense_fourier':
        from .highdim import make_dense_fourier
        return make_dense_fourier(d)
    if pde=='b6_fullrank':
        from .highdim import B6Equation,HJBActorBenchmark
        rows=json.loads(data_path('b6_frozen_parameters.json').read_text())
        row=next((r for r in rows if r['dimension']==d),None)
        if row is None: raise ValueError(f'B6 input does not define dimension {d}')
        return HJBActorBenchmark(B6Equation(d=d,beta=row['beta'],gamma=row['gamma']))
    raise KeyError(pde)


def is_joint(problem): return not hasattr(problem,'actor_kind')


def make_actor(problem, config: dict, device='cpu'):
    width,depth=int(config['width']),int(config['depth'])
    variant=config.get('variant','full')
    if is_joint(problem):
        if variant=='low_rank':
            actor=LearnedLowRankTerminalAwareActor(problem,width=width,depth=depth,rank=int(config.get('rank',2)),seed=int(config.get('seed',0))+11)
        elif variant=='no_terminal_anchor':
            from .ablation_models import UnanchoredJointActor
            actor=UnanchoredJointActor(problem,width=width,depth=depth)
        elif variant=='standard': actor=joint_models.StandardActor(problem,width=width,depth=depth)
        elif problem.family=='dense_fourier_joint':
            from .dense_actor import JointTerminalActor
            actor=JointTerminalActor(problem,width=width,depth=depth)
        else: actor=joint_models.TerminalAwareActor(problem,width=width,depth=depth)
    else:
        if variant=='low_rank': raise ValueError('Learned low-rank variant uses a joint value/gradient driver')
        cls=TerminalAwareActor
        if variant=='no_terminal_anchor':
            from .ablation_models import UnanchoredOriginalActor
            cls=UnanchoredOriginalActor
        elif variant=='standard': raise ValueError('StandardActor is the historical joint-only ablation')
        actor=cls(problem,width=width,depth=depth,activation=config.get('activation','silu'))
    return actor.to(torch.device(device))


def sample_states(problem,count,seed):
    if hasattr(problem,'sample_states'):
        purpose={'allen_cahn':'native','ou_gmm':'official'}.get(getattr(problem,'pde',''),'function')
        return problem.sample_states(count,seed=seed,purpose=purpose)
    rng=np.random.default_rng(seed)
    return rng.uniform(0.,np.nextafter(problem.T,0.),count),rng.standard_normal((count,problem.d))


def evaluate(problem,actor,t,x,*,paths=8,time_steps=4,seed=9001,device='cpu',parent_batch=4,path_chunk=4):
    fn=joint_evaluation.evaluate_policy if is_joint(problem) else fk.evaluate_policy
    values,meta=fn(problem,actor,t,x,paths=paths,time_steps=time_steps,seed=seed,device=torch.device(device),parent_batch=parent_batch,path_chunk=path_chunk)
    if is_joint(problem):
        sign=getattr(problem,'reconstruction_sign',1.)
        values['plain']=sign*values['plain'];values['mcv']=sign*values['mcv']
    return values,meta


def train(config: dict, output: str | Path):
    """Complete label -> fitting -> tangent-update -> paired-evaluation pipeline.

    Checkpoint selection uses noisy validation labels, never exact references.
    The seven-PDE suite can use its extracted reference-free cross-iterate
    selector. High-dimensional extension demonstrations explicitly use the
    final iterate; this does not reproduce their separate paper selector.
    """
    c=dict(config);seed=int(c.get('seed',0));device=torch.device(c.get('device','cpu'))
    if not 0<=seed<2**31: raise ValueError('seed must be in [0,2**31)')
    torch.set_num_threads(int(c.get('cpu_threads',2)));seed_everything(seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    problem=make_problem(c['pde'],int(c['dimension']));actor=make_actor(problem,c,device)
    started=time.perf_counter();history=[];candidates=[actor]
    keys=('training_points','validation_points','paths_per_label','label_time_steps','optimizer_steps','width','depth','learning_rate','batch_size','parent_batch','path_chunk','checkpoint_interval','gradient_clip')
    kwargs={k:c[k] for k in keys}
    joint=is_joint(problem)
    if joint:fit=joint_training.FitConfig(**kwargs)
    else:fit=fk.ActorConfig(**kwargs,activation=c.get('activation','silu'),training_distribution='function',label_estimator='plain' if c.get('variant')=='no_mcv' else 'mcv')
    updates=0 if c.get('variant')=='no_updates' else int(c.get('updates',2))
    estimator='plain' if c.get('variant')=='no_mcv' else 'mcv'
    for iteration in range(1,updates+1):
        tt,tx=sample_states(problem,fit.training_points,seed+1000*iteration+11)
        vt,vx=sample_states(problem,fit.validation_points,seed+29)
        common=dict(paths=fit.paths_per_label,time_steps=fit.label_time_steps,device=device,parent_batch=fit.parent_batch,path_chunk=fit.path_chunk)
        if joint:
            labeler=joint_fk.tangent_labels if estimator=='plain' else joint_mcv.mcv_tangent_labels
            tr,tq,tm=labeler(problem,actor,tt,tx,seed=seed+1000*iteration+101,**common)
            vr,vq,vm=labeler(problem,actor,vt,vx,seed=seed+1000*iteration+202,**common)
            if problem.family!='dense_fourier_joint':tr,tq,_=joint_fk.make_targets(problem,tr,tq);vr,vq,_=joint_fk.make_targets(problem,vr,vq)
            if c.get('variant')=='low_rank':
                arrays=dict(train_t=tt,train_x=tx,train_r=tr,train_q=tq,valid_t=vt,valid_x=vx,valid_r=vr,valid_q=vq)
                actor,meta,trace=train_low_rank_actor(problem,fit,int(c.get('rank',2)),arrays,seed=seed+iteration,device=device,initial=actor)
            elif c.get('variant')=='no_terminal_anchor':
                from .ablation_fitters import fit_joint
                actor,meta,trace=fit_joint(problem,fit,actor,lambda:make_actor(problem,c,device),tt,tx,tr,tq,vt,vx,vr,vq,seed=seed+iteration,device=device)
            elif problem.family=='dense_fourier_joint':
                from .dense_training import fit_joint_actor
                actor,meta,trace=fit_joint_actor(problem,fit,actor,tt,tx,tr,tq,vt,vx,vr,vq,seed=seed+iteration,device=device)
            else:
                actor,meta,trace=joint_training.train_actor(problem,fit,'standard' if c.get('variant')=='standard' else 'terminal_aware',tt,tx,tr,tq,vt,vx,vr,vq,seed=seed+iteration,device=device,initial=actor)
        else:
            tr,tm=fk.generate_labels(problem,actor,tt,tx,seed=seed+1000*iteration+101,estimator=estimator,**common)
            vr,vm=fk.generate_labels(problem,actor,vt,vx,seed=seed+1000*iteration+202,estimator=estimator,**common)
            if c.get('variant')=='no_terminal_anchor':
                from .ablation_fitters import fit_original
                actor,meta,trace=fit_original(problem,fit,actor,lambda:make_actor(problem,c,device),tt,tx,tr,vt,vx,vr,seed=seed+iteration,device=device)
            else: actor,meta,trace=fk.fit_actor(problem,fit,actor,tt,tx,tr,vt,vx,vr,seed=seed+iteration,device=device)
        history.append(dict(iteration=iteration,train_labels=tm,validation_labels=vm,fit=meta,trace=trace))
        save_checkpoint(out/f'actor_a{iteration}.pt',actor,c,iteration)
        candidates.append(actor)
    selection_rows=[];selected_iteration=updates
    if c.get('selection')=='reference_free':
        run_id=f"release_{c['pde']}_d{problem.d}_s{seed}"
        smoke=bool(c.get('selection_smoke',False))
        for i,candidate in enumerate(candidates):
            if joint:
                from .selection_joint import _fixed_point
                row=_fixed_point(run_id,f'a{i}',problem,candidate,training_seed=seed,device=device,smoke=smoke,estimator=estimator)
            else:
                from .selection_standard import _fixed_point
                row=_fixed_point(run_id,c['pde'],problem,f'a{i}',candidate,seed,device,smoke,estimator=estimator)
            selection_rows.append(row)
        eligible=[(i,row) for i,row in enumerate(selection_rows) if row['nonfinite_count']==0]
        if not eligible:raise FloatingPointError('No finite reference-free selection candidate')
        selected_iteration=min(eligible,key=lambda item:item[1]['normalized_total_inconsistency'])[0]
        actor=candidates[selected_iteration]
    et,ex=sample_states(problem,int(c.get('evaluation_states',8)),seed+90001)
    values,metrics=evaluate(problem,actor,et,ex,paths=int(c.get('evaluation_paths',8)),time_steps=int(c.get('evaluation_steps',4)),seed=seed+90002,device=device,parent_batch=fit.parent_batch,path_chunk=fit.path_chunk)
    save_checkpoint(out/'actor.pt',actor,c,selected_iteration)
    result=dict(pde=c['pde'],dimension=problem.d,updates=updates,selected_iteration=selected_iteration,variant=c.get('variant','full'),estimator=estimator,wall_seconds=time.perf_counter()-started,configuration=c,history=history,evaluation=metrics,output_mean_plain=float(np.mean(values['plain'])),output_mean_mcv=float(np.mean(values['mcv'])),finite=bool(all(np.isfinite(v).all() for v in values.values())),selection=c.get('selection','final'),selection_diagnostics=selection_rows,ground_truth_used_for_selection=False)
    result['primary_estimator']=estimator
    result['primary_output_mean']=float(np.mean(values[estimator]))
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'evaluation.npz',t=et,x=ex,**values)
    return result


def save_checkpoint(path,actor,config,iteration):
    torch.save({'release_schema':1,'state_dict':{k:v.detach().cpu() for k,v in actor.state_dict().items()},'configuration':dict(config),'iteration':int(iteration)},Path(path))


def load_checkpoint(path,*,device='cpu'):
    payload=torch.load(Path(path),map_location=device,weights_only=True)
    if payload.get('release_schema')!=1:raise ValueError('Import legacy checkpoint using scripts/import_checkpoint.py')
    c=payload['configuration'];problem=make_problem(c['pde'],int(c['dimension']));actor=make_actor(problem,c,device)
    actor.load_state_dict(payload['state_dict'],strict=True);actor.eval()
    return problem,actor,payload
