import torch
from torch import nn
from apt_dit.ddit import DDiT
from apt_dit.upstream import imports


def make():
    DiT,_=imports()
    base=DiT(input_size=32,patch_size=2,hidden_size=24,depth=1,num_heads=2)
    torch.nn.init.normal_(base.final_layer.linear.weight,std=.02)
    torch.nn.init.normal_(base.final_layer.adaLN_modulation[-1].weight,std=.02)
    for block in base.blocks:
        torch.nn.init.normal_(block.adaLN_modulation[-1].weight,std=.02)
    return DDiT(base)


def test_pseudoinverse_embedding():
    model=make()
    small=torch.randn(3,4,2,2)
    large=torch.nn.functional.interpolate(small,size=(4,4),mode='bilinear',align_corners=False)
    torch.testing.assert_close(model.patch4_embed(large),model.base.x_embedder.proj(small))


def test_geometry_and_mixed_coverage():
    model=make().eval()
    # Distinct channel/row/column codes distinguish pixel order from channel order.
    with torch.no_grad():
        model.base.final_layer.linear.weight.zero_()
        model.patch4_deembed.weight.zero_()
        model.base.final_layer.linear.bias.copy_(torch.arange(32))
        model.patch4_deembed.bias.copy_(1000+torch.arange(128))
    x=torch.randn(2,4,32,32); t=torch.tensor([0,999]); y=torch.tensor([1,1000])
    mask=torch.zeros(2,4,4,dtype=torch.bool)
    mask[0,::2,::2]=True; mask[1,1::2,1::2]=True
    out=model(x,t,y,coarse_blocks=mask)
    for n in range(2):
        for row in range(32):
            for col in range(32):
                p,offset=(4,1000) if mask[n,row//8,col//8] else (2,0)
                expected=offset+((row%p)*p+col%p)*8+torch.arange(8)
                torch.testing.assert_close(out[n,:,row,col],expected.float())
    fine=model(x,t,y)
    expected=model.base.unpatchify(model.base.final_layer.linear.bias.expand(2,256,-1))
    torch.testing.assert_close(fine,expected,rtol=0,atol=0)


def test_final_conditioning_and_frozen_teacher():
    model=make().eval()
    x=torch.randn(2,4,32,32);t=torch.tensor([10,800]);y=torch.tensor([1,1000])
    before=model(x,t,y).detach()
    seen=[]
    hook=model.base.final_layer.norm_final.register_forward_hook(lambda *args: seen.append(True))
    model(x,t,y,patch_size=4).sum().backward()
    hook.remove()
    assert len(seen)==2  # fine (empty) and coarse features both use final normalization
    assert all(p.grad is None for n,p in model.base.named_parameters() if 'parametrizations.weight.0.' not in n)
    with torch.no_grad():
        for p in model.trainable_parameters(): p.add_(.01)
    torch.testing.assert_close(model(x,t,y),before,rtol=0,atol=0)


def test_coarse_final_head_numerically_conditioned():
    model=make().eval(); x=torch.randn(2,4,32,32)
    t=torch.tensor([10,800]);y=torch.tensor([1,1000])
    received=[]
    hook=model.patch4_deembed.register_forward_pre_hook(lambda m,a:received.append(a[0].detach()))
    model(x,t,y,patch_size=4);hook.remove()
    b=model.base;c=model._condition(t,y)
    tokens=model.patch4_embed(x).flatten(2).transpose(1,2)+model.patch4_pos+model.patch4_id
    for block in b.blocks:tokens=block(tokens,c)
    shift,scale=b.final_layer.adaLN_modulation(c).chunk(2,1)
    expected=b.final_layer.norm_final(tokens)*(1+scale[:,None])+shift[:,None]
    torch.testing.assert_close(received[0],expected)
    assert not torch.allclose(expected,tokens)


def test_noise_skip_round_trip_and_fine_preservation():
    from apt_dit.ddit_recovery import NoiseSkipDDiT,coefficients
    DiT,_=imports(); model=NoiseSkipDDiT(DiT(input_size=32,patch_size=2,hidden_size=24,depth=1,num_heads=2))
    model.reset_correction();model.eval()
    x=torch.randn(2,4,32,32);t=torch.tensor([0,999]);y=torch.tensor([1,1000])
    alpha,sigma=coefficients(t,x)
    pred=model(x,t,y,patch_size=4)
    torch.testing.assert_close(pred[:,:4],sigma*x)
    torch.testing.assert_close(model(x,t,y),model.base(x,t,y),atol=0,rtol=0)
    target=torch.randn_like(x);v=(target-sigma*x)/alpha
    torch.testing.assert_close(sigma*x+alpha*v,target,atol=1e-6,rtol=1e-5)


def test_ddim_ignores_finite_variance_outputs():
    _,create=imports();diff=create('50')
    x=torch.randn(2,4,32,32);t=torch.tensor([49,20]);eps=torch.randn_like(x)
    a=diff.ddim_sample(lambda x,t:torch.cat([eps,torch.zeros_like(eps)],1),x,t,clip_denoised=False,eta=0.)
    b=diff.ddim_sample(lambda x,t:torch.cat([eps,torch.ones_like(eps)],1),x,t,clip_denoised=False,eta=0.)
    torch.testing.assert_close(a['sample'],b['sample'],atol=0,rtol=0)
