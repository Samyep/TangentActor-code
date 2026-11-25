"""Convert a trusted legacy checkpoint into the portable release schema."""
import argparse
from pathlib import Path
import torch
from tangent_actor import make_problem, make_actor, save_checkpoint


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--pde',required=True);p.add_argument('--dimension',required=True,type=int)
    p.add_argument('--variant',default='full',choices=['full','low_rank','no_terminal_anchor','standard'])
    a=p.parse_args();payload=torch.load(a.source,map_location='cpu',weights_only=True)
    c=dict(payload.get('config',payload.get('fit_config',{})))
    for k in ('width','depth','activation','rank'):
        if k in payload:c[k]=payload[k]
    c.update(pde=a.pde,dimension=a.dimension,variant=a.variant,seed=0)
    c.setdefault('activation','silu')
    problem=make_problem(a.pde,a.dimension);actor=make_actor(problem,c)
    actor.load_state_dict(payload['state_dict'],strict=True)
    out=Path(a.output)
    if out.exists():raise FileExistsError('Refusing to overwrite an existing checkpoint')
    out.parent.mkdir(parents=True,exist_ok=True)
    save_checkpoint(out,actor,c,payload.get('iteration',payload.get('metadata',{}).get('iteration',0)))
    print('Imported trusted checkpoint;strict state_dict load passed')


if __name__=='__main__':main()
