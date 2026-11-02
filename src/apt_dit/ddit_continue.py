"""Continue a versioned recovery checkpoint with held-out overfitting checks."""
import argparse
import json
import random
import time

import torch

from .ddit_recovery import (VERSION, NoiseSkipDDiT, load_pairs, objective,
                            validate, preview, fine_trajectory_check)
from .evaluate import CHECKPOINT, load_vae
from .runtime import GPUBudget, local_path, sha256, write_json
from .upstream import ROOT, load_dit


def restore_training(checkpoint, device='cuda'):
    if checkpoint.get('version') != VERSION or checkpoint.get('arm') != 'C':
        raise ValueError('This continuation expects a recovery-v1 C checkpoint')
    model=NoiseSkipDDiT(load_dit(CHECKPOINT),rank=32).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    params=model.trainable_parameters()
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=0.)
    optimizer.load_state_dict(checkpoint['optimizer'])
    rng=random.Random();rng.setstate(checkpoint['rng_state'])
    return model,params,optimizer,rng


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='artifacts/training/ddit_recovery_v1c/C/last.pt')
    p.add_argument('--reference-run',default='artifacts/training/ddit_recovery_v1c')
    p.add_argument('--name',default='ddit_recovery_long_c')
    p.add_argument('--additional-steps',type=int,default=20000)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--hours',type=float,default=.60)
    args=p.parse_args()
    if min(args.additional_steps,args.eval_every)<=0 or args.hours<=0:raise ValueError('Positive limits required')
    folder=local_path(f'artifacts/training/{args.name}')
    if folder.exists():raise FileExistsError(folder)
    source=local_path(args.checkpoint);previous=local_path(args.reference_run)
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    reference=torch.load(previous/'development_teacher.pt',map_location='cpu',weights_only=True)
    manifests=json.loads((previous/'manifest.json').read_text())
    train,train_meta=load_pairs('train');dev,dev_meta=load_pairs('validation')
    if train_meta!=manifests['train_records'] or dev_meta!=manifests['development_records']:
        raise ValueError('Cache records/order differ from the source experiment')
    entries=manifests['development_previews']
    start_step=checkpoint['step'];end_step=start_step+args.additional_steps
    folder.mkdir(parents=True)
    write_json(folder/'manifest.json',manifests)
    config=dict(version=VERSION,arm='C',resume=str(source.relative_to(ROOT)),resume_sha256=sha256(source),
        start_step=start_step,maximum_step=end_step,args=vars(args),learning_rate=1e-4,
        objective='v-target MSE; unchanged noise-skip architecture, paired draws, optimizer and data RNG',
        source_hashes={name:sha256(ROOT/f'src/apt_dit/{name}.py') for name in ('ddit','ddit_recovery','ddit_continue')},
        train_pairs=len(train[0]),development_pairs=len(dev[0]),
        train_evaluation='all training pairs at every check',development_evaluation='all held-out pairs',
        overfit_rule='three consecutive checks: development objective > best*1.05 while train objective falls',
        plateau_rule='five consecutive checks without at least 0.2% development-objective improvement',
        test_set_used=False)
    write_json(folder/'config.json',config)
    buckets=[torch.where(train[1][:,0]==v)[0].tolist() for v in sorted(train[1][:,0].unique().tolist())]
    with GPUBudget(args.name,args.hours) as budget:
        model,params,optimizer,rng=restore_training(checkpoint)
        vae=load_vae().cuda()
        import lpips
        metric=lpips.LPIPS(net='alex').eval().requires_grad_(False).cuda()
        parity=fine_trajectory_check(model)
        write_json(folder/'initial_fine_parity.json',parity)
        # Restore source RNG after setup. The source checkpoint precedes its
        # four-image periodic preview, which is replayed at the initial check.
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        best=float('inf');significant_best=float('inf');stale=0;overfit_checks=0
        previous_train=float('inf');logs=[];stop_reason='update_cap';started=time.monotonic()
        for step in range(start_step,end_step+1):
            budget.check()
            if step>start_step:
                ids=torch.tensor([rng.choice(buckets[(4*(step-1)+j)%len(buckets)]) for j in range(4)])
                x,t,y,target=[v[ids].flatten(0,1).cuda() for v in train]
                optimizer.zero_grad(set_to_none=True)
                loss=objective(model,'C',x,t,y,target)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at {step}')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
            if (step-start_step)%args.eval_every==0 or step==end_step:
                development=validate(model,dev,'C',budget)
                training=validate(model,train,'C',budget)
                score=development['objective'];train_score=training['objective']
                overfit_checks=overfit_checks+1 if score>best*1.05 and train_score<previous_train else 0
                previous_train=train_score
                improved=score<best
                if improved:best=score
                if score<significant_best*.998:significant_best=score;stale=0
                else:stale+=1
                row=dict(step=step,additional_updates=step-start_step,train=training,development=development,
                    overfit_checks=overfit_checks,plateau_checks=stale,seconds=time.monotonic()-started)
                if step>start_step:row.update(batch_loss=loss.item(),gradient_norm=norm.item())
                logs.append(row)
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=None,
                    rng_state=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    step=step,source_updates=checkpoint.get('source_updates',800),arm='C',version=VERSION,
                    config=config,validation=development,training=training)
                torch.save(state,folder/'last.pt')
                if improved:torch.save(state,folder/'best.pt')
                write_json(folder/'validation.json',logs)
                preview(model,vae,entries[:4],folder/f'preview_{step:05d}.png',budget)
                print(f'step={step} train_v={train_score:.7f} dev_v={score:.7f} '
                    f'train_eps={training["overall"]["epsilon_mse"]:.7f} dev_eps={development["overall"]["epsilon_mse"]:.7f} '
                    f'guided={development["overall"]["guided_mse"]:.7f} overfit_checks={overfit_checks}',flush=True)
                if overfit_checks>=3:stop_reason='sustained_overfitting';break
                if stale>=5:stop_reason='development_plateau';break
                # Leave time for final held-out images and parity checks.
                if budget.allowance-(time.monotonic()-budget.start)<60:
                    stop_reason='time_cap_reserve';break
        selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(selected['model'])
        parity=fine_trajectory_check(model)
        result=preview(model,vae,entries,folder/'best_development.png',budget,reference,metric)
        torch.save(result,folder/'best_development.pt')
        outcome=dict(start_step=start_step,last_step=step,best_step=selected['step'],stop_reason=stop_reason,
            initial=logs[0],best_development=selected['validation'],best_training=selected['training'],
            final=logs[-1],fine_parity=parity,lpips_mean=sum(result['lpips'])/len(result['lpips']),
            lpips_per_pair=result['lpips'],latent_abs_max=result['latent_abs_max'],
            seconds=time.monotonic()-started,test_set_used=False)
        write_json(folder/'outcome.json',outcome)
        write_json(folder/'status.json',dict(status='completed',stop_reason=stop_reason,best_step=selected['step']))
        print(f'completed: {stop_reason}, best step {selected["step"]}, LPIPS={outcome["lpips_mean"]:.5f}',flush=True)


if __name__=='__main__':main()
