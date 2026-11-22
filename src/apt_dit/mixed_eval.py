"""E2-E4 evaluation for the shared MixedPatchDiT.

Subcommands
  fidref      Inception (pytorch-fid) stats of the ADM ImageNet-256 reference batch
  calibrate   map tau -> mean tokens/step for temporal/spatial/joint on 64 images
  sweep       FID-5K (vs ImageNet ref and vs teacher), paired LPIPS (bootstrap CI),
              token profile, mask turnover for every arm in the H1/H2 grid
  gap         allocation gap: Spearman(score, LPIPS damage of coarsening one block)
  efficiency  executed FLOPs (FlopCounterMode) and measured latency
All evaluation seeds are fresh (>= 400000) and disjoint from training data.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg, stats as sstats
from torchvision.utils import save_image

from .mixed import load_mixed, tokens_for
from .mixed_sample import WARMUP, ddim_tables, initial_noise, sample_batch, window_scores
from .upstream import ROOT, load_dit

CKPT = ROOT / "artifacts/checkpoints/dit_s_256.pt"
OUT = ROOT / "artifacts/mixed_eval"
RES = ROOT / "results/mixed"


def jdump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def eval_set(n, offset=400_000):
    rng = random.Random(offset)
    return list(range(offset, offset + n)), [rng.randrange(1000) for _ in range(n)]


class Inception:
    def __init__(self):
        from pytorch_fid.inception import InceptionV3
        self.net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).cuda().eval()

    @torch.no_grad()
    def __call__(self, img01):
        return self.net(img01)[0].squeeze(-1).squeeze(-1).double().cpu()


def stats_of(feats):
    f = feats.numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def fid(s1, s2):
    mu1, sig1 = s1
    mu2, sig2 = s2
    covmean = linalg.sqrtm(sig1 @ sig2)
    if not np.isfinite(covmean).all():
        off = np.eye(sig1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sig1 + off) @ (sig2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


def fidref():
    path = OUT / "ref_stats.npz"
    if path.exists():
        return
    inc = Inception()
    arr = np.load(ROOT / "artifacts/fid/VIRTUAL_imagenet256_labeled.npz")["arr_0"]
    feats = []
    for lo in range(0, len(arr), 500):
        x = torch.from_numpy(arr[lo:lo + 500]).cuda().permute(0, 3, 1, 2).float() / 255
        feats.append(inc(x))
    mu, sig = stats_of(torch.cat(feats))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=mu, sigma=sig, n=len(arr))
    print(f"reference stats from {len(arr)} images", flush=True)


def ref_stats():
    d = np.load(OUT / "ref_stats.npz")
    return d["mu"], d["sigma"]


def load_models(ckpt):
    model, state = load_mixed(ckpt, CKPT)
    teacher = load_dit(CKPT, "cuda")
    return model, teacher, state


# ---------------------------------------------------------------- calibrate
@torch.no_grad()
def calibrate(args):
    model, teacher, state = load_models(args.checkpoint)
    lo, hi = state["tau_range"]
    taus = np.exp(np.linspace(math.log(lo) - 1, math.log(hi) + 1, args.points)).tolist()
    seeds, classes = eval_set(64, offset=500_000)
    table = {}
    for arm in ("temporal", "joint"):
        table[arm] = []
        for tau in taus:
            _, st = sample_batch(model, seeds, classes, arm=arm, tau=tau)
            table[arm].append(dict(tau=tau, tokens=st["tokens"].mean().item(),
                                   frac={p: float(np.mean(v)) for p, v in st["frac"].items()}))
            print(arm, f"tau={tau:.5f}", f"tokens={table[arm][-1]['tokens']:.1f}", flush=True)
    jdump(OUT / f"{args.tag}/calibration.json", table)


def tau_for(table, target):
    """Interpolate log tau giving mean tokens ~= target (tokens decrease with tau)."""
    pts = sorted(table, key=lambda r: r["tau"])
    for a, b in zip(pts, pts[1:]):
        if (a["tokens"] - target) * (b["tokens"] - target) <= 0 and a["tokens"] != b["tokens"]:
            w = (a["tokens"] - target) / (a["tokens"] - b["tokens"])
            return math.exp(math.log(a["tau"]) + w * (math.log(b["tau"]) - math.log(a["tau"])))
    return min(pts, key=lambda r: abs(r["tokens"] - target))["tau"]


# ---------------------------------------------------------------- sweep
@torch.no_grad()
def run_arm(model, teacher, vae, inc, seeds, classes, spec, batch, lpips_fn=None, ref_imgs=None, n_lpips=0):
    feats, lp, toks, turnover, fracs = [], [], [], [], []
    first = None
    t0 = time.time()
    for lo in range(0, len(seeds), batch):
        s, c = seeds[lo:lo + batch], classes[lo:lo + batch]
        if spec["arm"] == "teacher":
            lat, st = sample_batch(model, s, c, arm="fine", steps=spec["steps"], base=teacher)
        else:
            lat, st = sample_batch(model, s, c, arm=spec["arm"], tau=spec.get("tau"), k=spec.get("k", 0),
                                   random_seed=spec.get("rep", 0), target=spec.get("target"))
        toks.append(st["tokens"])
        turnover.append(np.mean(st["turnover"]) if st["turnover"] else 0.)
        fracs.append({p: float(np.mean(v)) for p, v in st["frac"].items()})
        for j in range(0, len(lat), 100):
            img = vae.decode(lat[j:j + 100] / .18215).sample.clamp(-1, 1)
            img01 = (img + 1) / 2
            # Quantize to uint8 like saved samples / the reference batch.
            img01 = (img01 * 255).round() / 255
            feats.append(inc(img01))
            g = lo + j
            if first is None:
                first = img[:64].cpu()
            if lpips_fn is not None and g < n_lpips:
                m = min(len(img), n_lpips - g)
                lp.append(lpips_fn(img[:m], ref_imgs[g:g + m].cuda().float()).flatten().cpu())
            elif ref_imgs is not None and lpips_fn is None and g < n_lpips:
                m = min(len(img), n_lpips - g)
                ref_imgs[g:g + m] = img[:m].half().cpu()
    tokens = torch.cat(toks)  # N x steps
    return dict(feats=torch.cat(feats), lpips=torch.cat(lp) if lp else None, tokens=tokens,
                turnover=float(np.mean(turnover)), first=first, seconds=time.time() - t0,
                frac={p: float(np.mean([f[p] for f in fracs])) for p in (2, 4, 8)})


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@torch.no_grad()
def sweep(args):
    import lpips
    from .evaluate import load_vae
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, teacher, state = load_models(args.checkpoint)
    vae = load_vae().cuda()
    inc = Inception()
    lpips_fn = lpips.LPIPS(net="alex").cuda().eval()
    ref = ref_stats()
    folder = OUT / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    cal = json.loads((folder / "calibration.json").read_text())
    seeds, classes = eval_set(args.n)
    n_lp = args.n_lpips
    ref_imgs = torch.zeros(n_lp, 3, 256, 256, dtype=torch.float16)
    specs = [dict(name="teacher50", arm="teacher", steps=50)]
    # Teacher seed-to-seed spread for the pre-registered matching tolerance delta.
    specs += [dict(name=f"teacher50_seedset{r}", arm="teacher", steps=50, seedset=r) for r in (1, 2)]
    specs += [dict(name=f"teacher{s}", arm="teacher", steps=s) for s in args.teacher_steps]
    specs += [dict(name="uniform4", arm="uniform4"), dict(name="uniform8", arm="uniform8")]
    specs += [dict(name=f"switch4_k{k}", arm="switch4", k=k) for k in args.switch_k]
    for target in args.targets:
        for fam in ("temporal", "spatial", "joint"):
            tau = tau_for(cal[fam], target) if fam != "spatial" else None
            specs.append(dict(name=f"{fam}_T{target}", arm=fam, tau=tau, target=target))
            if fam in ("spatial", "joint"):
                specs.append(dict(name=f"{fam}_random_T{target}", arm=f"{fam}_random", tau=tau, target=target))
    if args.only:
        specs = [s for s in specs if s["name"] in args.only or s["name"] == "teacher50"]
    results_path = folder / "sweep.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    teacher_feats = None
    for spec in specs:
        name = spec["name"]
        if name in results and name != "teacher50":
            print(f"skip {name} (done)", flush=True)
            continue
        if "seedset" in spec:
            s_, c_ = eval_set(args.n, offset=400_000 + 100_000 * spec["seedset"])
        else:
            s_, c_ = seeds, classes
        is_ref = name == "teacher50"
        r = run_arm(model, teacher, vae, inc, s_, c_, spec, args.batch,
                    lpips_fn=None if is_ref else lpips_fn, ref_imgs=ref_imgs,
                    n_lpips=0 if "seedset" in spec else n_lp)
        st = stats_of(r["feats"])
        if is_ref:
            teacher_feats = st
            torch.save(ref_imgs, folder / "teacher_ref_imgs.pt")
        elif teacher_feats is None:
            raise RuntimeError("teacher50 must run first")
        steps = spec.get("steps", 50)
        row = dict(spec, fid_imagenet=fid(st, ref), fid_teacher=0. if is_ref else fid(st, teacher_feats),
                   mean_tokens=float(r["tokens"].float().mean()), tokens_per_step=r["tokens"].float().mean(0).tolist(),
                   tokens_sq_per_step=r["tokens"].float().square().mean(0).tolist(),
                   forwards=steps, turnover=r["turnover"], frac=r["frac"], seconds=r["seconds"], n=len(s_))
        if r["lpips"] is not None:
            lp = r["lpips"].numpy()
            row.update(lpips=float(lp.mean()), lpips_ci=boot_ci(lp), n_lpips=len(lp))
            np.save(folder / f"lpips_{name}.npy", lp)
        results[name] = row
        save_image(r["first"], folder / f"samples_{name}.png", nrow=8, normalize=True, value_range=(-1, 1))
        jdump(results_path, results)
        print(f"{name}: FID-IN={row['fid_imagenet']:.2f} FID-T={row['fid_teacher']:.2f} "
              f"LPIPS={row.get('lpips', 0):.4f} tokens={row['mean_tokens']:.1f} ({r['seconds']:.0f}s)", flush=True)
        if is_ref:
            ref_imgs = torch.load(folder / "teacher_ref_imgs.pt")


# ---------------------------------------------------------------- allocation gap
def block_entropy(x0, bins=32):
    """Histogram entropy of the predicted clean latent within each 8x8 block (all channels)."""
    b = x0.reshape(-1, 4, 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5).flatten(3)  # B,4,4,256
    lo, hi = -3., 3.
    idx = ((b.clamp(lo, hi - 1e-6) - lo) / (hi - lo) * bins).long()
    hist = torch.zeros(*b.shape[:3], bins, device=x0.device).scatter_add_(-1, idx, torch.ones_like(b))
    p = hist / hist.sum(-1, keepdim=True)
    return -(p * (p + 1e-12).log()).sum(-1)


def block_variation(x0):
    b = x0.reshape(-1, 4, 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5)
    return (b.diff(dim=-1).abs().mean((3, 4, 5)) + b.diff(dim=-2).abs().mean((3, 4, 5))) / 2


@torch.no_grad()
def teacher_signals(teacher, seeds, classes, cfg=4.):
    """Run the fine teacher and accumulate per-block signals over scheduled steps."""
    x = initial_noise(seeds, "cuda")
    y = torch.as_tensor(classes, device="cuda")
    B = len(x)
    ab, abp, tmap = ddim_tables(50, "cuda")
    hist = [x]
    acc = dict(fd4=0, fd8=0, entropy=0, variation=0)
    n = 0
    for i, idx in enumerate(range(49, -1, -1)):
        t = torch.full((2 * B,), int(tmap[idx]), device="cuda")
        out = teacher(torch.cat([x, x]), t, torch.cat([y, torch.full_like(y, 1000)])).float()
        cond, uncond = out[:B, :4], out[B:, :4]
        eps = torch.cat([uncond[:, :3] + cfg * (cond[:, :3] - uncond[:, :3]), cond[:, 3:4]], 1)
        a, ap = ab[idx].item(), abp[idx].item()
        x0 = (1 / a) ** .5 * x - (1 / a - 1) ** .5 * eps
        if i >= WARMUP:
            d = hist[-1] - 3 * hist[-2] + 3 * hist[-3] - hist[-4]
            sc = window_scores(d)
            acc["fd4"] = acc["fd4"] + sc[4]
            acc["fd8"] = acc["fd8"] + sc[8]
            acc["entropy"] = acc["entropy"] + block_entropy(x0)
            acc["variation"] = acc["variation"] + block_variation(x0)
            n += 1
        eps2 = ((1 / a) ** .5 * x - x0) / (1 / a - 1) ** .5
        x = x0 * ap ** .5 + (1 - ap) ** .5 * eps2
        hist.append(x)
        hist = hist[-4:]
    return x, {k: (v / n) for k, v in acc.items()}


@torch.no_grad()
def gap(args):
    import lpips
    from .evaluate import load_vae
    from .mixed_sample import sample_batch as sb
    torch.backends.cuda.matmul.allow_tf32 = True
    model, teacher, _ = load_models(args.checkpoint)
    vae = load_vae().cuda()
    lpips_fn = lpips.LPIPS(net="alex").cuda().eval()
    seeds, classes = eval_set(args.n, offset=600_000)
    folder = OUT / args.tag
    out = {}
    for p in args.sizes:
        damages, sigs = [], {k: [] for k in ("fd4", "fd8", "entropy", "variation", "random")}
        for lo in range(0, len(seeds), args.batch):
            s, c = seeds[lo:lo + args.batch], classes[lo:lo + args.batch]
            ref_lat, sg = teacher_signals(teacher, s, c)
            ref_img = vae.decode(ref_lat / .18215).sample.clamp(-1, 1)
            dmg = torch.zeros(len(s), 16)
            for blk in range(16):
                lat = forced_block(model, s, c, blk, p)
                img = vae.decode(lat / .18215).sample.clamp(-1, 1)
                dmg[:, blk] = lpips_fn(img, ref_img).flatten().cpu()
            damages.append(dmg)
            for k in ("fd4", "fd8", "entropy", "variation"):
                sigs[k].append(sg[k].flatten(1).cpu())
            sigs["random"].append(torch.rand(len(s), 16, generator=torch.Generator().manual_seed(lo)))
            print(f"gap p={p}: {lo + len(s)}/{len(seeds)}", flush=True)
        D = torch.cat(damages).numpy()
        res = dict(mean_damage=float(D.mean()), damage_spread=float(D.std(1).mean()))
        for k, v in sigs.items():
            S = torch.cat(v).numpy()
            rhos = np.array([sstats.spearmanr(S[i], D[i]).statistic for i in range(len(D))])
            rhos = np.nan_to_num(rhos)
            lo_, hi_ = boot_ci(rhos)
            res[k] = dict(spearman=float(rhos.mean()), spearman_ci=[lo_, hi_],
                          gap=float((1 - rhos.mean()) / 2), gap_ci=[(1 - hi_) / 2, (1 - lo_) / 2])
            # Oracle-normalized: damage avoided when coarsening the k lowest-scored blocks.
        res["oracle_note"] = "gap=0 perfect ranking, 0.5 uninformative; per-image Spearman over 16 blocks"
        out[f"p{p}"] = res
        np.save(folder / f"gap_damage_p{p}.npy", D)
        print(json.dumps(res, indent=2), flush=True)
    jdump(folder / "gap.json", out)


@torch.no_grad()
def forced_block(model, seeds, classes, blk, p, cfg=4.):
    """Fine everywhere except one block forced to patch size p after warmup."""
    x = initial_noise(seeds, "cuda")
    y = torch.as_tensor(classes, device="cuda")
    B = len(x)
    ab, abp, tmap = ddim_tables(50, "cuda")
    lay = torch.full((B, 4, 4), 2, dtype=torch.long, device="cuda")
    lay.view(B, 16)[:, blk] = p
    for i, idx in enumerate(range(49, -1, -1)):
        t = torch.full((2 * B,), int(tmap[idx]), device="cuda")
        L = lay if i >= WARMUP else torch.full_like(lay, 2)
        out = model(torch.cat([x, x]), t, torch.cat([y, torch.full_like(y, 1000)]), torch.cat([L, L])).float()
        cond, uncond = out[:B, :4], out[B:, :4]
        eps = torch.cat([uncond[:, :3] + cfg * (cond[:, :3] - uncond[:, :3]), cond[:, 3:4]], 1)
        a, ap = ab[idx].item(), abp[idx].item()
        x0 = (1 / a) ** .5 * x - (1 / a - 1) ** .5 * eps
        eps2 = ((1 / a) ** .5 * x - x0) / (1 / a - 1) ** .5
        x = x0 * ap ** .5 + (1 - ap) ** .5 * eps2
    return x


# ---------------------------------------------------------------- efficiency
@torch.no_grad()
def efficiency(args):
    from torch.utils.flop_counter import FlopCounterMode
    model, teacher, _ = load_models(args.checkpoint)
    folder = OUT / args.tag
    res = dict(flops={}, latency=[])
    # Executed FLOPs of one CFG step (both branches) per token count.
    x = torch.randn(2, 4, 32, 32, device="cuda")
    t = torch.full((2,), 500, device="cuda")
    y = torch.tensor([1, 1000], device="cuda")
    combos = {}
    for n2 in range(17):
        for n4 in range(17 - n2):
            n8 = 16 - n2 - n4
            combos.setdefault(16 * n2 + 4 * n4 + n8, (n2, n4, n8))
    for n, (n2, n4, n8) in sorted(combos.items()):
        lay = torch.tensor([2] * n2 + [4] * n4 + [8] * n8, device="cuda").reshape(1, 4, 4).repeat(2, 1, 1)
        with FlopCounterMode(display=False) as fc:
            if n == 256:
                teacher(x, t, y)
            else:
                model(x, t, y, lay)
        res["flops"][n] = fc.get_total_flops()
    jdump(folder / "flops.json", res["flops"])

    def timed(fn, reps):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            st = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - st)
        return float(np.median(times)), float(np.min(times))

    cal = json.loads((folder / "calibration.json").read_text())
    configs = [("teacher50", dict(arm="fine", steps=50)), ("teacher25", dict(arm="fine", steps=25)),
               ("uniform4", dict(arm="uniform4")), ("uniform8", dict(arm="uniform8"))]
    for target in args.targets:
        for fam in ("temporal", "spatial", "joint"):
            configs.append((f"{fam}_T{target}", dict(arm=fam, target=target,
                                                      tau=None if fam == "spatial" else tau_for(cal[fam], target))))
    for dtype_name, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
        for bs in args.batch_sizes:
            seeds, classes = eval_set(bs, offset=700_000)
            for name, kw in configs:
                base = teacher if kw["arm"] == "fine" else None
                fn = lambda: sample_batch(model, seeds, classes, base=base, dtype=dtype, **kw)
                med, mn = timed(fn, args.reps if bs == 1 else 3)
                row = dict(config=name, dtype=dtype_name, batch=bs, median_s=med, min_s=mn,
                           images_per_s=bs / med)
                res["latency"].append(row)
                print(row, flush=True)
            # Trunk-only upper bound: one CFG forward at fixed token counts.
            for n2, n4, n8 in ((16, 0, 0), (8, 8, 0), (0, 16, 0), (0, 0, 16)):
                lay = torch.tensor([2] * n2 + [4] * n4 + [8] * n8, device="cuda").reshape(1, 4, 4).repeat(2 * bs, 1, 1)
                xx = torch.randn(2 * bs, 4, 32, 32, device="cuda")
                tt = torch.full((2 * bs,), 500, device="cuda")
                yy = torch.randint(0, 1000, (2 * bs,), device="cuda")
                with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
                    fn = (lambda: teacher(xx, tt, yy)) if n2 == 16 else (lambda: model(xx, tt, yy, lay))
                    med, mn = timed(fn, 20)
                row = dict(config=f"forward_tokens{16 * n2 + 4 * n4 + n8}", dtype=dtype_name, batch=bs,
                           median_s=med, min_s=mn)
                res["latency"].append(row)
                print(row, flush=True)
    jdump(folder / "efficiency.json", res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fidref", "calibrate", "sweep", "gap", "efficiency"])
    p.add_argument("--checkpoint", default=str(ROOT / "artifacts/training/mixed_lora32/last.pt"))
    p.add_argument("--tag", default="lora32")
    p.add_argument("--points", type=int, default=10)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--n-lpips", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--targets", type=int, nargs="+", default=[200, 150, 100])
    p.add_argument("--teacher-steps", type=int, nargs="+", default=[40, 33, 25, 20, 15, 10])
    p.add_argument("--switch-k", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--only", nargs="+")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 4])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 256])
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    (OUT / args.tag).mkdir(parents=True, exist_ok=True)
    dict(fidref=lambda: fidref(), calibrate=lambda: calibrate(args), sweep=lambda: sweep(args),
         gap=lambda: gap(args), efficiency=lambda: efficiency(args))[args.cmd]()


if __name__ == "__main__":
    main()
