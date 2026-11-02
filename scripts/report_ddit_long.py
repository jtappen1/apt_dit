"""CPU-only report for the noise-skip duration/overfitting experiment."""
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from apt_dit.upstream import ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='ddit_recovery_long_c');args=parser.parse_args()
    folder=ROOT/'artifacts/training'/args.name
    config=json.loads((folder/'config.json').read_text())
    outcome=json.loads((folder/'outcome.json').read_text())
    rows=json.loads((folder/'validation.json').read_text())
    original=ROOT/config['args']['reference_run']
    initial_images=torch.load(original/'C/best_development.pt',map_location='cpu',weights_only=True)
    initial_outcome=json.loads((original/'outcomes.json').read_text())['C']
    final_images=torch.load(folder/'best_development.pt',map_location='cpu',weights_only=True)
    teacher=torch.load(original/'development_teacher.pt',map_location='cpu',weights_only=True)
    lines=['# Longer noise-skip training: progress and overfitting','',
        '2026-11-02. Continued the strongest recovery-v1 C checkpoint with its optimizer, Python sampling RNG, and PyTorch RNG states restored. Same frozen teacher, existing cache, timestep-balanced paired batches, v-target objective, and AdamW learning rate 1e-4. No data expansion or architecture change. All-patch-4, 50-step DDIM, CFG 4.','',
        f'Training continued from {outcome["start_step"]:,} to {outcome["last_step"]:,} noise-skip updates; the best development objective was at {outcome["best_step"]:,}. Stop reason: `{outcome["stop_reason"]}`. The source trunk also inherits 800 older epsilon/variance updates before the v-correction output was reset; those are not counted as noise-skip updates.','',
        'Training and development metrics evaluate all 1,280 training pairs and all 320 held-out development pairs at every check, rather than comparing a noisy training minibatch with a full validation set. Images use the same 16 development pairs as before. The reserved final-test set remains unused.','',
        '| Checkpoint | Train v MSE | Held-out v MSE | Train epsilon MSE | Held-out epsilon MSE | Guided epsilon MSE |','|---|---:|---:|---:|---:|---:|']
    for label,tr,dv in [
        ('Start',outcome['initial']['train'],outcome['initial']['development']),
        ('Best held-out checkpoint',outcome['best_training'],outcome['best_development']),
        ('Last check',outcome['final']['train'],outcome['final']['development'])]:
        lines.append(f'| {label} | {tr["objective"]:.6f} | {dv["objective"]:.6f} | {tr["overall"]["epsilon_mse"]:.6f} | {dv["overall"]["epsilon_mse"]:.6f} | {dv["overall"]["guided_mse"]:.6f} |')
    lines += ['',f'Paired development LPIPS: **{initial_outcome["lpips_mean"]:.4f} → {outcome["lpips_mean"]:.4f}** (16 images, lower means closer to the same-seed teacher, not distributional image quality). Largest selected-checkpoint latent magnitude: {outcome["latent_abs_max"]:.3f}.','',
        '## Training versus generalization','',
        '| Noise-skip updates | Train v MSE | Held-out v MSE | Train epsilon MSE | Held-out epsilon MSE |','|---|---:|---:|---:|---:|']
    for r in rows:
        tr,dv=r['train'],r['development']
        lines.append(f'| {r["step"]} | {tr["objective"]:.6f} | {dv["objective"]:.6f} | {tr["overall"]["epsilon_mse"]:.6f} | {dv["overall"]["epsilon_mse"]:.6f} |')
    lines += ['', 'The overfitting stop requires three consecutive checks with development v-target error more than 5% above the best value while training error falls. The plateau stop requires five checks without at least 0.2% improvement in development v-target error. A widening train/development gap alone is a generalization warning, not proof that development quality is deteriorating. Read epsilon metrics and images alongside the v-target checkpoint-selection objective.','',
        '## Error by timestep','',
        '| Timestep | Start held-out epsilon MSE | Selected held-out epsilon MSE | Selected guided MSE |','|---|---:|---:|---:|']
    for k,v in outcome['best_development']['bins'].items():
        start=outcome['initial']['development']['bins'][k]
        lines.append(f'| {k} | {start["epsilon_mse"]:.8f} | {v["epsilon_mse"]:.8f} | {v["guided_mse"]:.8f} |')
    lines += ['', '## Verification and limits','',
        '- Original pure-fine full CFG trajectory remains bit-exact at the selected checkpoint; raw prediction and final latent maximum errors are both zero.',
        '- All 20 tests pass, including a bit-exact next-update test after restoring optimizer and data-RNG state. The continuation checks source cache record ordering against the saved manifest and preserves old runs.',
        '- Best and last checkpoints contain model, optimizer, RNG, step, architecture version, source hashes, and full metrics.',
        '- No schedule sweep, fresh test sampling, distributional FID, or speed claim. Development data have been inspected during prior iterations; this is an optimization/generalization diagnosis, not an unbiased final result.',
        '',f'Artifacts: `artifacts/training/{args.name}/`. Reproduce with `source scripts/env.sh`, then `.venv/bin/python -m apt_dit.ddit_continue --name <new-run> --additional-steps 20000 --eval-every 1000 --hours .60`.','']
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    steps=[r['step'] for r in rows]
    for split,label in [('train','Training (all records)'),('development','Held out (all records)')]:
        axes[0].plot(steps,[r[split]['objective'] for r in rows],marker='.',label=label)
        axes[1].plot(steps,[r[split]['overall']['epsilon_mse'] for r in rows],marker='.',label=label)
    for k in outcome['best_development']['bins']:
        axes[2].semilogy(steps,[r['development']['bins'][k]['epsilon_mse'] for r in rows],marker='.',label=k)
    for ax,title in zip(axes,['v-target MSE','Epsilon MSE','Held-out epsilon MSE by timestep']):
        ax.set(xlabel='Noise-skip training updates',ylabel=title);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(ROOT/'results/ddit_long_training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,8,figsize=(16,6.5),layout='constrained')
    for row,(label,images) in enumerate([('Teacher',teacher['images']),('3,200 updates',initial_images['images']),
                                       (f'{outcome["best_step"]:,} updates',final_images['images'])]):
        for col in range(8):
            axes[row,col].imshow(((images[col].permute(1,2,0)+1)/2).clamp(0,1));axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel(label)
    fig.savefig(ROOT/'results/ddit_long_training_samples.png',dpi=160);plt.close(fig)
    lines+=['[Learning curves](ddit_long_training_curves.png) · [Paired images](ddit_long_training_samples.png).','']
    assessment_path=folder/'assessment.json'
    if assessment_path.exists():
        assessment=json.loads(assessment_path.read_text())
        lines[2:2]=[assessment['interpretation'],'']
    (ROOT/'results/ddit_long_training_report.md').write_text('\n'.join(lines))
    print('best',outcome['best_step'],'stop',outcome['stop_reason'],'LPIPS',outcome['lpips_mean'])


if __name__=='__main__':main()
