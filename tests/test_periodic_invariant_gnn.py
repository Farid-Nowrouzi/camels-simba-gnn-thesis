import torch
from src.data.periodic_invariant_features import periodic_invariant_features
from src.models.periodic_invariant_gnn import PeriodicInvariantGNNRegressor,count_parameters
def graph():
    x=torch.tensor([[10.,1,1,1,0,0,0],[11.,2,1,1,1,0,0],[12.,3,1,1,2,0,0]],dtype=torch.float32,requires_grad=True); e=torch.tensor([[0,1,1,2],[1,0,2,1]]); return x,e
def test_count_shape_gradients_and_eval_determinism():
    x,e=graph(); n,a=periodic_invariant_features(x,e); m=PeriodicInvariantGNNRegressor(); assert count_parameters(m)==5270
    m.eval(); p=m(n,e,a,torch.zeros(3,dtype=torch.long),1); assert p.shape==(1,1) and torch.isfinite(p).all() and torch.equal(p,m(n,e,a,torch.zeros(3,dtype=torch.long),1))
    p.sum().backward(); assert all(p.grad is not None and torch.any(p.grad != 0) for p in m.parameters() if p.requires_grad)
def test_node_permutation_invariant():
    x,e=graph(); m=PeriodicInvariantGNNRegressor().eval(); n,a=periodic_invariant_features(x,e); p=m(n,e,a,torch.zeros(3,dtype=torch.long),1)
    perm=torch.tensor([2,0,1]); inverse=torch.argsort(perm); xp=x.detach()[perm]; ep=inverse[e]; nn,aa=periodic_invariant_features(xp,ep); assert torch.allclose(p,m(nn,ep,aa,torch.zeros(3,dtype=torch.long),1),atol=1e-6)
def test_end_to_end_cube_rotation_and_boost_invariance():
    x,e=graph(); m=PeriodicInvariantGNNRegressor().eval(); n,a=periodic_invariant_features(x,e); p=m(n,e,a,torch.zeros(3,dtype=torch.long),1)
    y=x.detach().clone(); rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    y[:,1:4]=(y[:,1:4]@rotation.T)%25; y[:,4:7]=y[:,4:7]@rotation.T+torch.tensor([8.,-2.,4.]); nn,aa=periodic_invariant_features(y,e)
    assert torch.allclose(p,m(nn,e,aa,torch.zeros(3,dtype=torch.long),1),atol=1e-6)
