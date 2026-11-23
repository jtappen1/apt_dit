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
from apt_dit import mixed_sample
from apt_dit.evaluate import load_vae
from apt_dit.mixed_eval import eval_set, boot_ci

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
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--cal-n', type=int, default=4)
    p.add_argument('--targets', nargs='+', type=int, default=[600])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.n >= 2 and args.cal_n >= 1
    args.output.mkdir(exist_ok=False, parents=True)
    base, model, step = load(args.checkpoint)
    print(f'Loaded trained XL/512 LoRA weights from update {step}', flush=True)
    vae = load_vae().eval().cuda()
    scorer = lpips.LPIPS(net='alex').eval().cuda()
    sample = sampler()
    seeds, classes = eval_set(args.n, 600_000)
    cal_seeds, cal_classes = eval_set(args.cal_n, 700_000)

    def denoise(s, c, **kw):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return sample(model, s, c, **kw)

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
                _, stats = denoise(cal_seeds, cal_classes, arm=arm, tau=tau)
                count = stats['tokens'].mean().item()
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
    for name, spec in specs:
        times, tokens, losses = [], [], []
        for i in range(args.n):
            torch.cuda.synchronize()
            start = time.perf_counter()
            latents, stats = denoise(seeds[i:i+1], classes[i:i+1], **spec)
            images = vae.decode(latents.float() / .18215).sample.float().clamp(-1, 1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            tokens.append(stats['tokens'].mean().item())
            if name == 'teacher50':
                reference.append(images.cpu())
            else:
                losses.append(scorer(images, reference[i].cuda()).item())
            if i < 4:
                previews.append(images.cpu())
        with FlopCounterMode(display=False) as counter:
            denoise(seeds[:1], classes[:1], **spec)
        row = dict(arm=name, n=args.n, tokens_mean=statistics.mean(tokens),
                   denoiser_counted_GFlops_one_pair=counter.get_total_flops() / 1e9,
                   median_end_to_end_seconds=statistics.median(times),
                   paired_lpips=statistics.mean(losses) if losses else 0.,
                   lpips_ci=list(map(float, boot_ci(__import__('numpy').array(losses)))) if losses else [0., 0.])
        print(json.dumps(row), flush=True)
        results.append(row)
        (args.output / 'metrics.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint), step=step,
            seeds=list(map(int, seeds)), classes=list(map(int, classes)), results=results), indent=2))
        save_image(torch.cat(previews), args.output / 'paired_samples.png', nrow=min(4, args.n), normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    main()
