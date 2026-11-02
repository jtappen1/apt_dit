"""Versioned, paired A/B/C recovery trials for the corrected patch-4 adapter."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from .ddit import DDiT
from .ddit_audit import metrics as spectral_metrics
from .ddit_evaluate import sample
from .evaluate import CHECKPOINT, load_vae
from .runtime import GPUBudget, local_path, manifest, sha256, write_json
from .upstream import ROOT, imports, load_dit

VERSION = 'ddit_recovery_v1'
SOURCE = 'artifacts/training/ddit_s_p4_corrected_slice/best.pt'


def coefficients(t, like):
    _, create = imports()
    values = torch.as_tensor(create('').alphas_cumprod, dtype=torch.float64, device=t.device)
    a = values[t.long()].to(like.dtype).reshape(-1, 1, 1, 1)
    return a.sqrt(), (1-a).sqrt()


class NoiseSkipDDiT(DDiT):
    """All-coarse v correction with original fine inference preserved exactly."""
    def __init__(self, base, rank=32):
        super().__init__(base, rank)
        _, create = imports()
        self.register_buffer('alpha_bar', torch.tensor(create('').alphas_cumprod, dtype=torch.float64), persistent=False)

    def reset_correction(self):
        with torch.no_grad():
            self.patch4_deembed.weight.reshape(4,4,8,-1)[:,:,:4].zero_()
            self.patch4_deembed.bias.reshape(4,4,8)[:,:,:4].zero_()
            self.residual.weight[:4].zero_()
            self.residual.bias[:4].zero_()

    def forward(self, x, t, y, patch_size=2, coarse_blocks=None, return_v=False):
        if coarse_blocks is not None and coarse_blocks.any() and not coarse_blocks.all():
            raise ValueError('Recovery v1 is trained for all-coarse inference only')
        raw = super().forward(x,t,y,patch_size=patch_size,coarse_blocks=coarse_blocks)
        adapted = patch_size == 4 if coarse_blocks is None else bool(coarse_blocks.any())
        if not adapted or return_v:
            return raw
        a = self.alpha_bar[t.long()].to(x.dtype).reshape(-1,1,1,1)
        eps = (1-a).sqrt()*x + a.sqrt()*raw[:,:4]
        return torch.cat([eps,raw[:,4:]],dim=1)


def load_pairs(split):
    rows=[]; provenance=[]
    for path in sorted(local_path(f'artifacts/cache/{split}').glob('*.pt')):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        for record in payload['records']:
            assert len(record['x'])==2 and torch.equal(record['x'][0],record['x'][1])
            assert record['y'][1].item()==1000 and record['t'][0]==record['t'][1]
            rows.append(tuple(record[k] for k in ('x','t','y','target')))
            provenance.append(dict(seed=payload['entry']['seed'],index=record['index'],timestep=int(record['t'][0])))
    if not rows: raise FileNotFoundError(split)
    return tuple(torch.stack([r[i] for r in rows]) for i in range(4)),provenance


def build(arm, source=SOURCE):
    torch.manual_seed(6220)
    cls=NoiseSkipDDiT if arm=='C' else DDiT
    model=cls(load_dit(CHECKPOINT),rank=32)
    model.load_state_dict(torch.load(local_path(source),map_location='cpu',weights_only=True)['model'],strict=True)
    if arm=='C': model.reset_correction()
    return model.eval()


def objective(model,arm,x,t,y,target):
    if arm=='C':
        raw=model(x,t,y,patch_size=4,return_v=True)
        alpha,sigma=coefficients(t,x)
        truth=(target[:,:4]-sigma*x)/alpha
        return (raw[:,:4]-truth).square().mean()
    pred=model(x,t,y,patch_size=4)
    return (pred-target).square().mean() if arm=='A' else (pred[:,:4]-target[:,:4]).square().mean()


def guided(eps):
    """Paired [pairs,2,4,H,W] -> retained conditional CFG trajectory epsilon."""
    return torch.cat([eps[:,1,:3]+4*(eps[:,0,:3]-eps[:,1,:3]),eps[:,0,3:4]],dim=1)


@torch.no_grad()
def validate(model,data,arm,budget=None,limit=None):
    ids=torch.arange(len(data[0]))
    if limit is not None: ids=ids[torch.linspace(0,len(ids)-1,min(limit,len(ids))).round().long()]
    preds=[]; losses=[]
    for selected in ids.split(4):
        if budget: budget.check()
        x,t,y,target=[v[selected].flatten(0,1).cuda() for v in data]
        pred=model(x,t,y,patch_size=4)
        preds.append(pred.cpu().reshape(len(selected),2,8,32,32))
        losses.append((len(selected),objective(model,arm,x,t,y,target).item()))
    pred=torch.cat(preds); x,t,y,target=[v[ids] for v in data]
    err=guided(pred[:,:,:4])-guided(target[:,:,:4])
    _,create=imports(); diff=create('50'); mapping={v:i for i,v in enumerate(diff.timestep_map)}
    pos=torch.tensor([mapping[int(v)] for v in t[:,0]])
    a=torch.tensor(diff.alphas_cumprod)[pos]; ap=torch.tensor(diff.alphas_cumprod_prev)[pos]
    coef=((1-ap).sqrt()-(ap/a).sqrt()*(1-a).sqrt()).float()
    def summarize(mask):
        pp,tt=pred[mask],target[mask]
        return dict(epsilon_mse=(pp[:,:,:4]-tt[:,:,:4]).square().mean().item(),
            conditional_mse=(pp[:,0,:4]-tt[:,0,:4]).square().mean().item(),
            unconditional_mse=(pp[:,1,:4]-tt[:,1,:4]).square().mean().item(),
            variance_mse=(pp[:,:,4:]-tt[:,:,4:]).square().mean().item(),
            guided_mse=err[mask].square().mean().item(),
            next_state_mse=(err[mask]*coef[mask,None,None,None]).square().mean().item(),
            identity_mse=(x[mask]-tt[:,:,:4]).square().mean().item())
    result=dict(objective=sum(n*l for n,l in losses)/len(ids),pairs=len(ids),overall=summarize(torch.ones(len(ids),dtype=torch.bool)),bins={})
    for lo in range(0,1000,200):
        mask=(t[:,0]>=lo)&(t[:,0]<lo+200)
        if mask.any():result['bins'][f'{lo}-{lo+199}']=summarize(mask)
    result['spectral']=spectral_metrics(pred.flatten(0,1),target.flatten(0,1))
    return result


@torch.no_grad()
def fine_trajectory_check(model):
    _,create=imports(); diffusion=create('50')
    z=torch.randn(2,4,32,32,device='cuda',generator=torch.Generator(device='cuda').manual_seed(7721))
    labels=torch.tensor([1,1000],device='cuda'); max_error=0.
    def predictor(x,t):
        nonlocal max_error
        combined=torch.cat([x[:1],x[:1]])
        raw=model(combined,t,labels,patch_size=2)
        eps=raw[:,:3]; g=eps[1:]+4*(eps[:1]-eps[1:])
        actual=torch.cat([torch.cat([g,g]),raw[:,3:]],dim=1)
        expected=model.base.forward_with_cfg(x,t,labels,4.)
        max_error=max(max_error,(actual-expected).abs().max().item())
        return actual
    actual=diffusion.ddim_sample_loop(predictor,z.shape,noise=z.clone(),clip_denoised=False,device='cuda',eta=0.)
    expected=diffusion.ddim_sample_loop(model.base.forward_with_cfg,z.shape,noise=z.clone(),clip_denoised=False,device='cuda',eta=0.,model_kwargs=dict(y=labels,cfg_scale=4.))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    return dict(max_prediction_error=max_error,final_latent_max_error=(actual-expected).abs().max().item())


@torch.no_grad()
def preview(model,vae,entries,path,budget,reference=None,metric=None):
    images=[];latents=[];scores=[]
    for entry in entries:
        latent,_=sample(model,**entry,strategy='all_coarse',budget=budget)
        if not torch.isfinite(latent).all(): raise FloatingPointError('Nonfinite preview latent')
        images.append(vae.decode(latent/.18215).sample.clamp(-1,1).cpu());latents.append(latent.cpu())
    image=torch.cat(images); latent=torch.cat(latents)
    save_image(image,path,nrow=min(8,len(entries)),normalize=True,value_range=(-1,1))
    if metric is not None:
        for i in range(len(image)):
            scores.append(metric(image[i:i+1].cuda(),reference['images'][i:i+1].cuda()).item())
    return dict(images=image,latents=latent,lpips=scores,latent_abs_max=latent.abs().max().item())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=3200)
    p.add_argument('--name',default='ddit_recovery_v1')
    p.add_argument('--tiny-steps',type=int,default=400)
    p.add_argument('--hours',type=float,default=.60)
    args=p.parse_args()
    if args.steps<1 or args.tiny_steps<1 or args.hours<=0:raise ValueError('Positive limits required')
    folder=local_path(f'artifacts/training/{args.name}')
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    data,train_prov=load_pairs('train'); dev,dev_prov=load_pairs('validation')
    entries=manifest()['validation'][:16]
    rng_manifest=random.Random(91822)
    write_json(folder/'manifest.json',dict(train=manifest()['train'],development=manifest()['validation'],
        development_previews=entries,test_reserved=[dict(seed=40000+i,class_id=rng_manifest.randrange(1000)) for i in range(64)],
        train_records=train_prov,development_records=dev_prov,note='Test reserved before training; not sampled or inspected in this experiment'))
    config=dict(version=VERSION,source=SOURCE,source_sha256=sha256(local_path(SOURCE)),teacher_sha256=sha256(CHECKPOINT),
        source_code_sha256={name:sha256(ROOT/f'src/apt_dit/{name}.py') for name in ('ddit','ddit_recovery','ddit_evaluate')},
        additional_updates=args.steps,learning_rate=1e-4,weight_decay=0.,batch_pairs=4,seed=6220,
        checkpoint_selection='minimum arm-specific development objective',
        arms=dict(A='warm-start corrected; all-eight-channel MSE',B='same warm-start; epsilon MSE only',C='warm-start trunk; zero correction; v-target MSE'),
        notes=['A/B use new optimizers; not exact legacy resume','A/B/C share timestep-balanced paired draw order','C changes both initialization and parameterization/objective','all-coarse training and evaluation only'],args=vars(args))
    write_json(folder/'config.json',config)
    buckets=[torch.where(data[1][:,0]==v)[0].tolist() for v in sorted(data[1][:,0].unique().tolist())]
    with GPUBudget(args.name,args.hours) as budget:
        vae=load_vae().cuda()
        import lpips
        metric=lpips.LPIPS(net='alex').eval().requires_grad_(False).cuda()
        baseline=build('A').cuda().requires_grad_(False)
        checks=fine_trajectory_check(baseline)
        # Teacher-only development reference, generated before selecting any arm.
        from .sampling import sample as teacher_sample
        reference_images=[];reference_latents=[]
        for entry in entries:
            z,_,_=teacher_sample(baseline.base,**entry,budget=budget)
            reference_images.append(vae.decode(z/.18215).sample.clamp(-1,1).cpu());reference_latents.append(z.cpu())
        reference=dict(images=torch.cat(reference_images),latents=torch.cat(reference_latents))
        save_image(reference['images'],folder/'development_teacher.png',nrow=8,normalize=True,value_range=(-1,1))
        torch.save(reference,folder/'development_teacher.pt')
        write_json(folder/'fine_parity.json',checks)
        del baseline
        # Tiny optimization check is isolated; none of these weights enter A/B/C.
        tiny=build('C').cuda(); params=tiny.trainable_parameters();opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0.)
        tiny_ids=torch.tensor([buckets[i][0] for i in (0,5,9)])
        xx,tt,yy,target=[v[tiny_ids].flatten(0,1).cuda() for v in data]
        initial=objective(tiny,'C',xx,tt,yy,target).item()
        for step in range(args.tiny_steps):
            budget.check();opt.zero_grad(set_to_none=True);loss=objective(tiny,'C',xx,tt,yy,target)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            if (step+1)%100==0:
                check=objective(tiny,'C',xx,tt,yy,target).item()
                print(f'tiny step={step+1} loss={check:.7f}',flush=True)
                if check<initial/100:break
        final=objective(tiny,'C',xx,tt,yy,target).item()
        write_json(folder/'tiny_fit.json',dict(initial=initial,final=final,reduction=initial/final,steps=step+1,
            timesteps=tt.cpu().tolist(),learning_rate=1e-3,required_reduction=100,passed=final<initial/100))
        print(f'tiny fit {initial:.6f} -> {final:.6f}, reduction {initial/final:.1f}x',flush=True)
        del tiny,opt,params
        if final>=initial/100:
            write_json(folder/'status.json',dict(status='tiny_fit_failed',reason='Do not expand cache or launch broader training until optimization check passes'))
            return
        outcomes={}
        for arm in ('A','B','C'):
            directory=folder/arm;directory.mkdir()
            model=build(arm).cuda();params=model.trainable_parameters()
            optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
            rng=random.Random(6220);torch.manual_seed(6220);torch.cuda.manual_seed_all(6220)
            logs=[];best=float('inf');stale=0;started=time.monotonic()
            for step in range(args.steps+1):
                budget.check()
                if step:
                    # Cycle over the ten timestep strata; random trajectories within each.
                    ids=torch.tensor([rng.choice(buckets[(4*(step-1)+j)%len(buckets)]) for j in range(4)])
                    x,t,y,target=[v[ids].flatten(0,1).cuda() for v in data]
                    optimizer.zero_grad(set_to_none=True);loss=objective(model,arm,x,t,y,target)
                    if not torch.isfinite(loss):raise FloatingPointError(f'{arm} step {step}')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                if step%400==0 or step==args.steps:
                    score=validate(model,dev,arm,budget)
                    train_score=validate(model,data,arm,budget,limit=64)
                    row=dict(step=step,development=score,train=train_score,seconds=time.monotonic()-started)
                    if step:row.update(batch_loss=loss.item(),gradient_norm=norm.item())
                    logs.append(row)
                    improved=score['objective']<best*(1-1e-4)
                    if improved:best=score['objective'];stale=0
                    else:stale+=1
                    checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,
                        rng_state=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                        step=step,source_updates=800,arm=arm,version=VERSION,config=config,validation=score)
                    torch.save(checkpoint,directory/'last.pt')
                    if improved:torch.save(checkpoint,directory/'best.pt')
                    write_json(directory/'validation.json',logs)
                    preview(model,vae,entries[:4],directory/f'preview_{step:04d}.png',budget)
                    print(f'{arm} +{step}: objective={score["objective"]:.6f}, eps={score["overall"]["epsilon_mse"]:.6f}, guided={score["overall"]["guided_mse"]:.6f}, high={score["bins"]["800-999"]["epsilon_mse"]:.8f}',flush=True)
                    if stale>=2:
                        print(f'{arm}: stopped after two checks without improvement',flush=True);break
            state=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state['model'])
            parity=fine_trajectory_check(model)
            result=preview(model,vae,entries,directory/'best_development.png',budget,reference,metric)
            torch.save(result,directory/'best_development.pt')
            outcome=dict(best_step=state['step'],validation=state['validation'],fine_parity=parity,
                lpips_mean=sum(result['lpips'])/len(entries),lpips_per_pair=result['lpips'],
                latent_abs_max=result['latent_abs_max'],seconds=time.monotonic()-started)
            outcomes[arm]=outcome;write_json(folder/'outcomes.json',outcomes)
            del model,optimizer,params,state,checkpoint
        write_json(folder/'status.json',dict(status='completed',arms=list(outcomes),test_set_used=False))


if __name__=='__main__':main()
