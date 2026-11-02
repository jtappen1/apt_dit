"""Reproducible held-out audit; writes only new diagnostic evidence."""
import argparse
import numpy as np
import torch
from .ddit import DDiT
from .ddit_train import load_split
from .evaluate import CHECKPOINT, load_vae
from .runtime import GPUBudget, local_path, write_json, manifest
from .upstream import load_dit


def metrics(pred, target):
    result = {}
    for name, sl in [('epsilon', slice(0,4)), ('variance', slice(4,8))]:
        a, b = pred[:,sl].double(), target[:,sl].double()
        err = a-b
        spectrum = torch.fft.fft2(err, norm='ortho').abs().square()
        truth = torch.fft.fft2(b, norm='ortho').abs().square()
        freq = torch.fft.fftfreq(32)
        high = torch.maximum(freq[:,None].abs(), freq[None,:].abs()) >= .25
        result[name] = dict(mse=err.square().mean().item(), target_variance=b.var(unbiased=False).item(),
            low_frequency_mse=spectrum[...,~high].mean().item(), high_frequency_mse=spectrum[...,high].mean().item(),
            high_frequency_target_energy=truth[...,high].mean().item())
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--name', default='ddit_audit_initial')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--split', choices=['train','validation'], default='validation')
    parser.add_argument('--legacy-head', action='store_true')
    args=parser.parse_args()
    torch.manual_seed(6220)
    folder=local_path('results')
    with GPUBudget(args.name, .25) as budget:
        if args.checkpoint:
            from .ddit_evaluate import load_model
            model=load_model(args.checkpoint).cuda()
            if args.legacy_head and model.__class__ is not DDiT:
                raise ValueError('Legacy-head audit requires a historical DDiT checkpoint')
        else:
            model=DDiT(load_dit(CHECKPOINT)).cuda().eval()
        data=load_split(args.split)
        p2_teacher_max=0.
        predictions=[]; references=[]
        for ids in torch.arange(len(data[0])).split(8):
            budget.check()
            x,t,y,target=[v[ids].cuda() for v in data]
            references.append(model(x,t,y,patch_size=2).cpu())
            p2_teacher_max=max(p2_teacher_max,(model.base(x,t,y).cpu()-references[-1]).abs().max().item())
            hooks=[]
            if args.legacy_head:
                hooks=[model.base.final_layer.norm_final.register_forward_hook(lambda m,a,o:a[0]),
                       model.base.final_layer.adaLN_modulation.register_forward_hook(lambda m,a,o:torch.zeros_like(o))]
            predictions.append(model(x,t,y,patch_size=4).cpu())
            for hook in hooks: hook.remove()
        pred=torch.cat(predictions); ref=torch.cat(references); target=data[3]
        report=dict(records=len(pred), split=args.split, legacy_head=args.legacy_head, p2_teacher_max_abs=p2_teacher_max, p2_cache_max_abs=(ref-target).abs().max().item(),
                    p2_cache_mse=(ref-target).square().mean().item(), overall=metrics(pred,target), bins={})
        for lo,hi in [(0,200),(200,400),(400,600),(600,800),(800,1000)]:
            mask=(data[1]>=lo)&(data[1]<hi)
            if mask.any(): report['bins'][f'{lo}-{hi-1}']=dict(records=int(mask.sum()), **metrics(pred[mask],target[mask]))
        report['prediction_gate']=all(v['mse'] < v['target_variance'] for v in report['overall'].values()) and all(
            v[k]['mse'] < v[k]['target_variance'] for v in report['bins'].values() for k in ('epsilon','variance'))
        np.savez_compressed(folder/f'{args.name}.npz', prediction=pred[:8].numpy(), teacher=target[:8].numpy(),t=data[1][:8].numpy())
        write_json(folder/f'{args.name}.json', report)
        print(report,flush=True)
        if args.preview:
            from .ddit_evaluate import sample
            from torchvision.utils import save_image
            vae=load_vae().cuda()
            images=[]
            for entry in manifest()['final'][:4]:
                latent,_=sample(model,**entry,strategy='all_coarse',budget=budget)
                images.append(vae.decode(latent/.18215).sample.clamp(-1,1).cpu())
            save_image(torch.cat(images),folder/f'{args.name}_p4.png',nrow=4,normalize=True,value_range=(-1,1))


if __name__=='__main__': main()
