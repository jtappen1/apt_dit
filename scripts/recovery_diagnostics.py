"""CPU-only planning diagnostic on existing validation caches; no model execution."""
import json
import torch
from apt_dit.ddit_train import load_split
from apt_dit.upstream import ROOT, imports


def main():
    torch.set_num_threads(4)
    x,t,y,target=load_split('validation')
    _,create_diffusion=imports()
    diffusion=create_diffusion('50')
    mapping={v:k for k,v in enumerate(diffusion.timestep_map)}
    ids=torch.tensor([mapping[int(v)] for v in t])
    a=torch.tensor(diffusion.alphas_cumprod)[ids].reshape(-1,1,1,1)
    ap=torch.tensor(diffusion.alphas_cumprod_prev)[ids].reshape(-1,1,1,1)
    identity=x.double()
    scaled=identity/(1-a).sqrt()
    teacher=target[:,:4].double()
    result=dict(records=len(x),note='CPU cache analysis only. Identity baselines are diagnostics, not usable denoisers.',bins={})
    for lo,hi in [(0,200),(200,400),(400,600),(600,800),(800,1000)]:
        m=(t>=lo)&(t<hi)
        result['bins'][f'{lo}-{hi-1}']=dict(
            epsilon_equals_x_mse=(identity[m]-teacher[m]).square().mean().item(),
            epsilon_equals_scaled_x_mse=(scaled[m]-teacher[m]).square().mean().item(),
            teacher_epsilon_variance=teacher[m].var(unbiased=False).item(),
            ddim_epsilon_error_coefficient_min=(torch.sqrt(1-ap[m])-torch.sqrt(ap[m]/a[m])*torch.sqrt(1-a[m])).abs().min().item(),
            ddim_epsilon_error_coefficient_max=(torch.sqrt(1-ap[m])-torch.sqrt(ap[m]/a[m])*torch.sqrt(1-a[m])).abs().max().item(),
            x0_epsilon_error_coefficient_max=torch.sqrt((1-a[m])/a[m]).max().item())
    arr=torch.load(ROOT/'artifacts/training/ddit_s_p4_corrected_slice/best.pt',map_location='cpu',weights_only=True)
    w=arr['model']['residual.weight'].double()
    result['corrected_residual']=dict(epsilon_center_matrix=w[:4,:,1,1].tolist(),epsilon_weight_norm=w[:4].norm().item(),variance_weight_norm=w[4:].norm().item())
    result['learning_curves']={}
    for p in sorted((ROOT/'artifacts/training').glob('*/validation.json')):
        rows=json.loads(p.read_text()); result['learning_curves'][p.parent.name]=dict(first=rows[0],last=rows[-1],checks=len(rows))
    ledger=json.loads((ROOT/'artifacts/gpu_ledger.json').read_text())
    used=sum(v.get('seconds',v['reserved_seconds']) for v in ledger)
    result['gpu_ledger']=dict(charged_hours=used/3600,remaining_hours=max(0,8-used/3600))
    path=ROOT/'results/ddit_recovery_diagnostics.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    # Reuse the existing model audit; no forward pass is performed here.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    audit=json.loads((ROOT/'results/ddit_audit_corrected.json').read_text())
    labels=list(result['bins'])
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.semilogy(labels,[result['bins'][k]['epsilon_equals_x_mse'] for k in labels],
                'o-',label='Input latent as epsilon (CPU baseline)')
    ax.semilogy(labels,[audit['bins'][k]['epsilon']['mse'] for k in labels],
                'o-',label='Corrected patch-4 model (saved audit)')
    ax.set(xlabel='Original diffusion timestep bin (higher = noisier)',
           ylabel='Teacher epsilon MSE (log scale)',
           title='The patch-4 predictor loses an easy high-noise signal')
    ax.grid(alpha=.25); ax.legend()
    fig.savefig(ROOT/'results/ddit_recovery_diagnostics.png',dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in result.items() if k!='learning_curves'},indent=2))

if __name__=='__main__': main()
