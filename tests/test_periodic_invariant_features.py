import torch
from src.data.periodic_invariant_features import periodic_invariant_features
def sample(): return torch.tensor([[10.,.1,0,0,0,0,0],[11.,24.9,0,0,-100.,0,0.]],dtype=torch.float32),torch.tensor([[0],[1]])
def test_order_and_outward_radial_sign():
    x,e=sample(); node,attr=periodic_invariant_features(x,e); assert node.shape==(2,1) and attr.shape==(1,3)
    assert torch.allclose(attr[0],torch.tensor([.008,.1,.1]),atol=1e-6) # displacement -.2 and dv -100 => outward
def test_common_boost_translation_and_reflection_invariant():
    x,e=sample(); _,a=periodic_invariant_features(x,e); y=x.clone(); y[:,1:4]=(y[:,1:4]+7)%25; y[:,4:7]+=33
    _,b=periodic_invariant_features(y,e); assert torch.allclose(a,b)
    y=x.clone(); y[:,1:4]=(-y[:,1:4])%25; y[:,4:7]*=-1; _,b=periodic_invariant_features(y,e); assert torch.allclose(a,b)
def test_invalid_edges_rejected():
    x,_=sample()
    for bad_edge in [torch.tensor([0,1]),torch.tensor([[0],[0]]),torch.tensor([[0],[2]])]:
        try: periodic_invariant_features(x,bad_edge)
        except (ValueError,TypeError): continue
        raise AssertionError("expected malformed edge rejection")
def test_bad_width_rejected():
    try: periodic_invariant_features(torch.zeros(2,6),torch.tensor([[0],[1]]))
    except ValueError: return
    raise AssertionError("expected width rejection")
