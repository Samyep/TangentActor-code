import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tangent_actor import PDES, make_problem, make_actor, train, evaluate, load_checkpoint, data_path
from tangent_actor.api import is_joint
from tangent_actor import feynman_kac as fk, joint_fk, joint_mcv

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def config(pde='hjb_log'):
    c=json.loads((ROOT/'configs/smoke_hjb_log.json').read_text())
    c.update(pde=pde,optimizer_steps=6,checkpoint_interval=3)
    return c


@pytest.mark.parametrize('pde',PDES)
def test_terminal_condition(pde):
    d=10 if pde=='ou_gmm' else 20 if pde=='b6_fullrank' else 4
    problem=make_problem(pde,d);actor=make_actor(problem,dict(config(pde),dimension=d))
    with torch.no_grad():
        for p in actor.parameters():p.add_(.07)
    x=torch.randn(3,d);t=torch.full((3,),problem.T)
    if is_joint(problem):
        r,q=actor.predict(t,x,problem.T)
        torch.testing.assert_close(r,problem.terminal(x),rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(q,problem.sigma*problem.terminal_gradient_torch(x),rtol=2e-6,atol=2e-6)
    else:
        r,q=actor.predict(t,x)
        if problem.actor_kind=='gradient':torch.testing.assert_close(q,problem.sigma*problem.terminal_gradient(x))
        else:torch.testing.assert_close(r,problem.terminal(x))


@pytest.mark.parametrize('pde',PDES)
def test_tangent_linearization_identity(pde):
    d=10 if pde=='ou_gmm' else 20 if pde=='b6_fullrank' else 4
    problem=make_problem(pde,d);r=torch.rand(5)*.2+.4;q=torch.randn(5,d)*.1
    if not is_joint(problem) and problem.actor_kind=='value':q=None
    c,b,h=problem.tangent_coefficients(r,q)
    reconstructed=c*r+h+(0 if q is None else (b*q).sum(-1))
    torch.testing.assert_close(reconstructed,problem.driver(r,q),rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('pde',['hjb_log','joint_concave','nonconvex_coupled','dense_fourier'])
def test_fk_mcv_finite_outputs_and_pathwise_labels(pde):
    problem=make_problem(pde,4);actor=make_actor(problem,config(pde));t=np.array([.02,.08]);x=np.ones((2,4))*.1
    values,meta=evaluate(problem,actor,t,x,paths=8,time_steps=4,parent_batch=2,path_chunk=4)
    assert all(np.isfinite(v).all() for v in values.values());assert meta['antithetic_pairs']==4
    kwargs=dict(paths=8,time_steps=4,seed=17,device=torch.device('cpu'),parent_batch=2,path_chunk=4)
    if is_joint(problem):
        r,z,_=joint_mcv.mcv_tangent_labels(problem,actor,t,x,**kwargs);assert np.isfinite(r).all()
    else:z,_=fk.generate_labels(problem,actor,t,x,estimator='mcv',**kwargs)
    assert z.shape==(2,4) and np.isfinite(z).all()


def test_mcv_is_paired_on_identical_brownian_paths():
    p=make_problem('hjb_log',3);a=make_actor(p,config());t=torch.tensor([.1,.3]);x=torch.randn(2,3)
    normals=fk.antithetic_normals(2,8,4,3,generator=torch.Generator().manual_seed(19),device=torch.device('cpu'))
    torch.testing.assert_close(normals[:,:4],-normals[:,4:])
    plain0,corrected0,w0,_,_=fk._rollout_chunk(p,a,t,x,normals,mcv=False,need_gradient=False)
    plain1,corrected1,w1,_,_=fk._rollout_chunk(p,a,t,x,normals,mcv=True,need_gradient=False)
    torch.testing.assert_close(plain0,plain1);torch.testing.assert_close(w0,w1);torch.testing.assert_close(plain0,corrected0)
    assert not torch.equal(plain1,corrected1)
    v,_=fk.evaluate_policy(p,a,t.numpy(),x.numpy(),paths=8,time_steps=4,seed=19,device=torch.device('cpu'),parent_batch=2,path_chunk=8)
    pair=.5*(corrected1[:,:4]+corrected1[:,4:])
    np.testing.assert_allclose(v['mcv'],pair.double().mean(1).numpy(),rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(v['mcv_pair_variance'],pair.double().var(1,unbiased=True).numpy(),rtol=2e-6,atol=2e-6)


def test_projection_and_learned_low_rank_keep_full_ambient_input():
    p=make_problem('nonconvex_coupled',5);a=make_actor(p,dict(config(),variant='low_rank',rank=2))
    assert a.network[0].in_features==6 and a.gradient_basis.shape==(5,2) and a.gradient_basis.requires_grad
    z=np.ones((2,5));r,z2,m=joint_fk.make_targets(p,np.array([-1.,2.]),z)
    assert np.all((r>=p.r_low)&(r<=p.r_high));np.testing.assert_array_equal(z,z2);assert m['value_projection_rate']==1.


def test_standard_and_unanchored_variants_are_real_distinct_models():
    p=make_problem('joint_concave',4);t=torch.full((2,),p.T);x=torch.zeros(2,4)
    for variant in ('standard','no_terminal_anchor'):
        a=make_actor(p,dict(config(),variant=variant))
        with torch.no_grad():a.network[-1].bias.add_(.2)
        r,_=a.predict(t,x,p.T)
        assert not torch.allclose(r,p.terminal(x))


@pytest.fixture(autouse=True)
def local_inputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    np.savez(tmp_path / "ou_gmm_d10.npz", dimension=10,
             mean=rng.normal(size=(5, 10)), pi=np.full(5, 0.2))
    (tmp_path / "b6_frozen_parameters.json").write_text(json.dumps([
        {"dimension": 20, "beta": 1.0, "gamma": 0.5}
    ]))
    monkeypatch.setenv("TANGENT_ACTOR_DATA_DIR", str(tmp_path))


def test_user_supplied_data():
    assert make_problem("ou_gmm", 10).mean.shape == (5, 10)
    with pytest.raises(FileNotFoundError):
        data_path("no-such-input.npz")


@pytest.mark.parametrize('pde',['hjb_log','nonconvex_coupled'])
def test_end_to_end_labels_fit_updates_selection_save_load_evaluate(pde,tmp_path):
    c=config(pde);report=train(c,tmp_path/pde)
    assert report['finite'] and len(report['history'])==2
    assert len(report['selection_diagnostics'])==3 and not report['ground_truth_used_for_selection']
    assert all(h['train_labels']['ground_truth_used'] is False if 'ground_truth_used' in h['train_labels'] else h['train_labels']['analytic_nonterminal_reference_used'] is False for h in report['history'])
    problem,actor,payload=load_checkpoint(tmp_path/pde/'actor.pt')
    assert payload['iteration']==report['selected_iteration']
    values,_=evaluate(problem,actor,np.array([.05]),np.ones((1,problem.d))*.2)
    assert np.isfinite(values['mcv']).all()


@pytest.mark.parametrize('variant',['low_rank','no_terminal_anchor','no_mcv','no_updates','standard'])
def test_variants_have_runnable_training_or_fixed_policy(variant,tmp_path):
    c=config('nonconvex_coupled');c.update(variant=variant,updates=1,selection='final',optimizer_steps=3,checkpoint_interval=1)
    r=train(c,tmp_path/variant);assert r['finite'];assert r['updates']==(0 if variant=='no_updates' else 1)


@pytest.mark.parametrize('pde',PDES)
def test_every_pde_completes_one_update(pde,tmp_path):
    d=10 if pde=='ou_gmm' else 20 if pde=='b6_fullrank' else 4
    c=config(pde)
    c.update(dimension=d,updates=1,training_points=8,validation_points=4,
             optimizer_steps=2,checkpoint_interval=1,evaluation_states=2,
             selection='final' if pde in ('b6_fullrank','dense_fourier') else 'reference_free')
    r=train(c,tmp_path/pde)
    assert r['finite'] and len(r['history'])==1
    assert r['history'][0]['fit']


@pytest.mark.parametrize('pde',['hjb_log','nonconvex_coupled'])
def test_no_mcv_selector_and_primary_output_use_plain_estimator(pde,tmp_path):
    c=config(pde);c.update(variant='no_mcv',updates=1,optimizer_steps=2,checkpoint_interval=1)
    r=train(c,tmp_path/pde)
    assert r['primary_estimator']=='plain'
    assert r['primary_output_mean']==r['output_mean_plain']
    assert all(row['estimator']=='plain' for row in r['selection_diagnostics'])
