import torch
from src.data.periodic_geometry import signed_minimum_image_displacement as delta
def test_boundary_reverse_translation_dtype_and_inputs():
    s=torch.tensor([0.1,0.,0.],dtype=torch.float64); t=torch.tensor([24.9,0.,0.],dtype=torch.float64); saved=s.clone()
    assert torch.allclose(delta(s,t,25.),torch.tensor([-0.2,0.,0.],dtype=torch.float64)); assert torch.allclose(delta(t,s,25.),-delta(s,t,25.)); assert s.equal(saved)
    assert torch.allclose(delta(s+50,t-25,25.),delta(s,t,25.)); assert delta(s,t,25.).dtype is torch.float64
def test_half_box_is_torch_round_half_even():
    assert delta(torch.zeros(3),torch.tensor([12.5,0.,0.]),25.)[0].item()==12.5
    assert delta(torch.zeros(3),torch.tensor([37.5,0.,0.]),25.)[0].item()==-12.5
def test_bad_box_rejected():
    try: delta(torch.zeros(3),torch.zeros(3),0)
    except ValueError: return
    raise AssertionError("expected ValueError")
