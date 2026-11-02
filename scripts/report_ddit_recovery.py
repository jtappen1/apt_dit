"""Summarize the completed controlled recovery runs without new GPU work."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
from apt_dit.upstream import ROOT
from apt_dit.runtime import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='ddit_recovery_v1c');args=parser.parse_args()
    folder=ROOT/'artifacts/training'/args.name
    outcomes=json.loads((folder/'outcomes.json').read_text())
    tiny=json.loads((folder/'tiny_fit.json').read_text())
    curves={a:json.loads((folder/a/'validation.json').read_text()) for a in outcomes}
    reference=torch.load(folder/'development_teacher.pt',map_location='cpu',weights_only=True)
    images=[reference['images'][:8]]
    names=dict(A='Longer corrected training',B='Longer, epsilon-only',C='Explicit noise skip + v correction')
    lines=['# Controlled DiT-S recovery: noise skip and longer training','',
      '2026-11-02. These are development experiments on the existing held-out validation trajectories, not fresh final-test results. All arms use 50-step deterministic DDIM, CFG 4 with upstream three-channel guidance, and all-patch-4 inference. Pure patch-2 behavior remains exact. Original artifacts are preserved.','',
      '## Setup','',
      'A and B warm-start the same corrected 800-update checkpoint and reset AdamW. Each receives up to 3,200 additional updates with identical timestep-balanced paired sample draws, batch size four conditional/unconditional pairs, learning rate 1e-4, and frozen DiT weights. A minimizes all-eight-channel MSE; B minimizes epsilon MSE. C warm-starts the same trunk but zeros its correction outputs and predicts v with an explicit full-resolution noise skip. C changes initialization, parameterization, and objective together, so it is not a single-factor comparison with B.','',
      '`eps = sqrt(1-alpha_bar) * x + sqrt(alpha_bar) * v`; teacher targets use the exact inverse conversion. Learned variance remains an eight-channel interface component but is excluded from B/C training objectives. No mixed-layout or scheduler hypothesis is tested.','',
      f'The isolated C optimization check reduced loss from {tiny["initial"]:.6f} to {tiny["final"]:.6f} ({tiny["reduction"]:.1f}x) in {tiny["steps"]} updates at learning rate 1e-3. Its weights are discarded before the controlled runs.','',
      'Checkpoint selection uses each arm’s own minimum development objective. Metrics below use all 320 held-out pairs (640 branches). Image LPIPS uses the same 16 development seeds/classes for each arm and a freshly generated teacher reference. The new 64-pair final-test manifest remains unused.','',
      '| Arm | Selected additional updates | Epsilon MSE | Guided epsilon MSE | DDIM next-state MSE | Development LPIPS (16) |','|---|---:|---:|---:|---:|---:|']
    for a,o in outcomes.items():
        m=o['validation']['overall']
        lines.append(f'| {a}: {names[a]} | {o["best_step"]} | {m["epsilon_mse"]:.6f} | {m["guided_mse"]:.6f} | {m["next_state_mse"]:.6f} | {o["lpips_mean"]:.4f} |')
        record=torch.load(folder/a/'best_development.pt',map_location='cpu',weights_only=True)
        images.append(record['images'][:8])
    save_image(torch.cat(images),ROOT/'results/ddit_recovery_samples.png',nrow=8,normalize=True,value_range=(-1,1))
    lines += ['', 'Image rows: teacher, A, B, C. Columns share class and seed. See [paired images](ddit_recovery_samples.png). LPIPS measures same-seed fidelity, not distributional quality.','',
        '| Timestep bin | Identity epsilon MSE | A epsilon MSE | B epsilon MSE | C epsilon MSE |','|---|---:|---:|---:|---:|']
    for k,v in outcomes['A']['validation']['bins'].items():
        values=[outcomes[a]['validation']['bins'][k]['epsilon_mse'] for a in outcomes]
        lines.append(f'| {k} | {v["identity_mse"]:.8f} | '+' | '.join(f'{v:.8f}' for v in values)+' |')
    lines += ['', '## Does longer training help?', '']
    for a in outcomes:
        start=curves[a][0]['development']['overall'];end=outcomes[a]['validation']['overall']
        lines.append(f'- {a}: epsilon MSE {start["epsilon_mse"]:.6f} → {end["epsilon_mse"]:.6f}; guided epsilon MSE {start["guided_mse"]:.6f} → {end["guided_mse"]:.6f}.')
    lines += ['', 'This separates a measured improvement in prediction error from image recognition, which still requires inspection of the saved contact sheet. A/B isolate loss allocation with matched updates. C tests the proposed noise-preservation package. Improvement over initialization is not evidence of convergence or acceptable images.','',
        '## Validation and limitations','',
        '- CPU suite: 19 tests passed before training, including nonzero final-head conditioning, v conversion, and deterministic-DDIM variance independence.',
        '- Complete pure fine CFG trajectories are checked against pinned upstream before training and for each selected checkpoint. Both raw predictions and final latent are required to be bit-exact.',
        '- Full conditional, unconditional, guided, next-state, spectral, and timestep metrics are retained in `outcomes.json` and each arm’s `validation.json`. The next-state metric is same-state error, not full-rollout error.',
        '- `best.pt` and `last.pt` include optimizer and RNG state, arm/version, source hashes, initialization, and objective. Warm-start histories are explicitly distinguished from exact optimizer resume.',
        '- No cache expansion, final-test sampling, schedule sweeps, or FID run. Existing development data were reused and have been inspected; results are not an unbiased final estimate.',
        '- No speed claim is made. The new architecture retains dense fine embeddings and residual work, and LoRA remains unmerged.',
        '',f'Artifacts: `artifacts/training/{args.name}/`. Reproduce with `source scripts/env.sh`, then `.venv/bin/python -m apt_dit.ddit_recovery --name <new-run> --steps 3200 --tiny-steps 1200 --hours .60`. This command creates a new immutable run directory.','']
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for a,rows in curves.items():
        steps=[r['step'] for r in rows]
        for ax,key in zip(axes,('epsilon_mse','guided_mse')):
            ax.semilogy(steps,[r['development']['overall'][key] for r in rows],marker='o',label=a)
        axes[2].semilogy(steps,[r['development']['bins']['800-999']['epsilon_mse'] for r in rows],marker='o',label=a)
    for ax,title in zip(axes,('Epsilon MSE','Guided epsilon MSE','High-noise epsilon MSE')):
        ax.set(xlabel='Additional updates',ylabel=title);ax.legend();ax.grid(alpha=.2)
    fig.savefig(ROOT/'results/ddit_recovery_curves.png',dpi=160);plt.close(fig)
    lines+=['[Learning curves](ddit_recovery_curves.png).','']
    report_text='\n'.join(lines)
    assessment_path=ROOT/'results/ddit_recovery_assessment.json'
    if assessment_path.exists():
        assessment=json.loads(assessment_path.read_text())
        report_text=report_text.replace('## Setup',assessment.get('report_interpretation','')+'## Setup',1)
        seconds=assessment['recovery_gpu_process_seconds']
        report_text+=f'\nRecovery GPU-process time, including guarded preliminary checks: {seconds:.1f} seconds ({seconds/3600:.4f} hours), charged in the existing ledger.\n'
    (ROOT/'results/ddit_recovery_report.md').write_text(report_text)
    write_json(ROOT/'results/ddit_recovery_summary.json',outcomes)
    print('\n'.join(lines[:22]))


if __name__=='__main__':main()
