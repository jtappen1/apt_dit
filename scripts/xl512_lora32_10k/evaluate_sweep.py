import argparse
import importlib.util
import inspect
import json
from pathlib import Path
import statistics
import time
import types

import torch
from torch.utils.flop_counter import FlopCounterMode
from torchvision.utils import save_image
import lpips
import numpy as np
from apt_dit import mixed_sample
from apt_dit.evaluate import load_vae
from apt_dit.mixed_eval import eval_set, boot_ci, Inception, stats_of, fid

RUN = Path('runs/xl512_lora32_10k')


def sampler():
    module = types.ModuleType('apt_dit.sampler_xl512')
    module.__dict__.update({k: v for k, v in vars(mixed_sample).items() if not k.startswith('__')})
    changes = {
        'budget_layout': [('WARMUP * 256', 'WARMUP * 1024'), ('256 - T', '1024 - T'),
                          ('T >= 64', 'T >= 256'), ('64 - T', '256 - T'), ('16', '64'), ('4, 4', '8, 8')],
        'reassign': [('4, 4', '8, 8')],
        'permute_blocks': [('16', '64'), ('4, 4', '8, 8')],
        'initial_noise': [('4, 32, 32', '4, 64, 64')],
        'sample_batch': [('4, 4)', '8, 8)')],
    }
    for name, pairs in changes.items():
        source = inspect.getsource(getattr(mixed_sample, name))
        for old, new in pairs:
            assert old in source, (name, old)
            source = source.replace(old, new)
        exec(source, module.__dict__)
    return module.sample_batch


def load(checkpoint):
    spec = importlib.util.spec_from_file_location('train_xl512', Path(__file__).with_name('train.py'))
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    base, model = trainer.load_model()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(weights['adapter']) == expected and weights['resolution'] == 512
    model.load_state_dict(weights['adapter'], strict=False)
    return base.eval(), model.eval(), weights['step']


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=(Path('artifacts/xl512_lora32_10k/best.pt') if Path('artifacts/xl512_lora32_10k/best.pt').exists() else RUN / 'best.pt'))
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--n-lpips', type=int, default=1000)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--fid-reference-stats', type=Path)
    p.add_argument('--cal-n', type=int, default=16)
    p.add_argument('--targets', nargs='+', type=int, default=[800, 600, 400])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.n >= 2 and args.cal_n >= 1
    args.output.mkdir(exist_ok=False, parents=True)
    base, model, step = load(args.checkpoint)
    print(f'Loaded trained XL/512 LoRA weights from update {step}', flush=True)
    vae = load_vae().eval().cuda()
    scorer = lpips.LPIPS(net='alex').eval().cuda()
    sample = sampler()
    inception = Inception() if args.n >= 2049 else None
    if args.fid_reference_stats and inception is None:
        p.error('Use at least 2049 generated samples for this FID sweep')
    if args.fid_reference_stats:
        with np.load(args.fid_reference_stats) as data:
            real_stats = data['mu'], data['sigma']
    else:
        real_stats = None
    teacher_stats = None
    n_lpips = min(args.n_lpips, args.n)
    assert args.batch > 0 and n_lpips > 0
    seeds, classes = eval_set(args.n, 600_000)
    cal_seeds, cal_classes = eval_set(args.cal_n, 700_000)

    def denoise(s, c, **kw):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return sample(model, s, c, dtype=torch.bfloat16, **kw)

    specs = [('teacher50', dict(arm='fine', base=base)), ('teacher25', dict(arm='fine', steps=25, base=base)),
             ('uniform4', dict(arm='uniform4')), ('uniform8', dict(arm='uniform8')),
             ('switch20', dict(arm='switch4', k=20))]
    for target in args.targets:
        specs += [(f'{arm}_T{target}', dict(arm=arm, target=target)) for arm in ('spatial', 'spatial_random')]
        for arm in ('temporal', 'joint'):
            options = []
            low, high = -6., -1.5
            for _ in range(8):
                middle = (low + high) / 2
                tau = 10 ** middle
                counts = []
                for i in range(0, len(cal_seeds), args.batch):
                    sl = slice(i, i + args.batch)
                    _, stats = denoise(cal_seeds[sl], cal_classes[sl], arm=arm, tau=tau)
                    counts.append(stats['tokens'].flatten())
                count = torch.cat(counts).mean().item()
                options.append((abs(count - target), tau, count))
                if count > target:
                    low = middle
                else:
                    high = middle
            _, tau, count = min(options)
            print(f'Calibration {arm} target={target} achieved={count:.1f} tau={tau:.6g}', flush=True)
            specs.append((f'{arm}_T{target}', dict(arm=arm, tau=tau)))
            if arm == 'joint':
                specs.append((f'joint_random_T{target}', dict(arm='joint_random', tau=tau)))

    # Warm up the sampler and decoder before recording per-image latency.
    warm, _ = denoise(seeds[:1], classes[:1], arm='fine', steps=5, base=base)
    vae.decode(warm.float() / .18215).sample
    reference, results, previews = [], [], []
    specs[1:2] = [(f'teacher{s}', dict(arm='fine', steps=s, base=base)) for s in (40, 33, 25, 20, 15, 10)]
    specs[9:10] = [(f'switch{k}', dict(arm='switch4', k=k)) for k in (10, 20, 30)]
    for name, spec in specs:
        times, tokens, losses, feats = [], [], [], []
        print(f'Generating {name}: {args.n} images', flush=True)
        for i in range(0, args.n, args.batch):
            torch.cuda.synchronize()
            start = time.perf_counter()
            sl = slice(i, i + args.batch)
            if spec['arm'].endswith('_random'):
                latents, stats = denoise(seeds[sl], classes[sl], random_seed=i, **spec)
            else:
                latents, stats = denoise(seeds[sl], classes[sl], **spec)
            images = vae.decode(latents.float() / .18215).sample.float().clamp(-1, 1)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) / len(images))
            tokens.extend(stats['tokens'].mean(dim=1).tolist())
            if inception is not None:
                quantized = (((images + 1) / 2) * 255).round() / 255
                feats.append(inception(quantized))
            n_pair = max(0, min(n_lpips - i, len(images)))
            if name == 'teacher50':
                reference.extend(images[:n_pair].to(device='cpu', dtype=torch.float16).unbind())
            elif n_pair:
                refs = torch.stack(reference[i:i+n_pair]).to(device='cuda', dtype=torch.float32)
                losses.extend(scorer(images[:n_pair], refs).flatten().cpu().tolist())
            if i < 4:
                previews.append(images[:4-i].cpu())
            if (i // args.batch) % 25 == 0:
                print(f'{name}: {i + len(images)}/{args.n}', flush=True)
        with FlopCounterMode(display=False) as counter:
            denoise(seeds[:1], classes[:1], **spec)
        row = dict(arm=name, n=args.n, tokens_mean=statistics.mean(tokens),
                   denoiser_counted_GFlops_one_pair=counter.get_total_flops() / 1e9,
                   median_batched_seconds_per_image=statistics.median(times), sampling_batch=args.batch,
                   paired_lpips=statistics.mean(losses) if losses else 0.,
                   lpips_ci=list(map(float, boot_ci(__import__('numpy').array(losses)))) if losses else [0., 0.])
        if inception is not None:
            features = stats_of(torch.cat(feats))
            if name == 'teacher50':
                teacher_stats = features
            row['fid_teacher'] = 0. if name == 'teacher50' else fid(features, teacher_stats)
            if real_stats is not None:
                row['fid_imagenet_512'] = fid(features, real_stats)
        print(json.dumps(row), flush=True)
        results.append(row)
        (args.output / 'metrics.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint), step=step,
            seeds=list(map(int, seeds)), classes=list(map(int, classes)), results=results), indent=2))
        save_image(torch.cat(previews), args.output / 'paired_samples.png', nrow=min(4, args.n), normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    main()
