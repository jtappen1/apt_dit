"""A resumed optimizer must produce the same next update, not just load weights."""
import copy
import random

import torch

from apt_dit.ddit_recovery import NoiseSkipDDiT, VERSION, objective
from apt_dit.upstream import imports
from apt_dit import ddit_continue


def test_optimizer_and_draw_state_resume(monkeypatch):
    DiT,_=imports()
    def base():
        return DiT(input_size=32,patch_size=2,hidden_size=24,depth=1,num_heads=2)
    monkeypatch.setattr(ddit_continue,'load_dit',lambda _:base())
    torch.manual_seed(91)
    model=NoiseSkipDDiT(base()).eval();model.reset_correction()
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=1e-4,weight_decay=0.)
    x=torch.randn(2,4,32,32);t=torch.tensor([100,800]);y=torch.tensor([1,1000]);target=torch.randn(2,8,32,32)
    def update(m,opt):
        opt.zero_grad(set_to_none=True)
        objective(m,'C',x,t,y,target).backward();opt.step()
    update(model,optimizer)
    rng=random.Random(62);rng.randrange(128)
    checkpoint=copy.deepcopy(dict(version=VERSION,arm='C',model=model.state_dict(),
                                optimizer=optimizer.state_dict(),rng_state=rng.getstate()))
    restored,_,resumed_optimizer,resumed_rng=ddit_continue.restore_training(checkpoint,device='cpu')
    assert rng.randrange(128)==resumed_rng.randrange(128)
    update(model,optimizer);update(restored,resumed_optimizer)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
