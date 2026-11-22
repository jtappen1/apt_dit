"""Self-distill one shared MixedPatchDiT from the frozen DiT-S teacher.

Data (proposal Sec. 3.3, no external images):
  * N teacher CFG-4 samples (final latents x0), classes uniform over all 1000;
  * a subset of full 50-step teacher trajectories, so training can use the
    real DDIM states and the real Delta^3 selector masks seen at inference.
Each batch: half trajectory states (with selector-derived or random layouts),
half x0 re-noised at uniform t in [0, 999]. 10% of labels are the null class
so CFG's unconditional branch is distilled too. Targets: teacher outputs.
Layout mix: uniform-4, uniform-8, smooth random fields over {2,4,8},
and Eq. 3 selector masks at random thresholds (trajectory states only).
"""
import argparse
import json
import math
import random
import time

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from .mixed import MixedPatchDiT
from .mixed_sample import WARMUP, choose, ddim_tables, sample_batch, window_scores
from .upstream import ROOT, imports, load_dit

CKPT = ROOT / "artifacts/checkpoints/dit_s_256.pt"
DATA = ROOT / "artifacts/mixed_data"


@torch.no_grad()
def generate(teacher, n_samples, n_traj, batch, seed_offset=1_000_000):
    """Teacher CFG-4 50-step DDIM samples. Seeds disjoint from evaluation."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"teacher_{n_samples}_{n_traj}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    rng = random.Random(6220)
    classes = torch.tensor([rng.randrange(1000) for _ in range(n_samples)])
    seeds = torch.arange(n_samples) + seed_offset
    x0s, trajs = [], []
    ab, abp, tmap = ddim_tables(50, "cuda")
    start = time.time()
    for lo in range(0, n_samples, batch):
        s, c = seeds[lo:lo + batch], classes[lo:lo + batch]
        want = lo < n_traj
        x = torch.stack([torch.randn(4, 32, 32, generator=torch.Generator("cuda").manual_seed(int(v)), device="cuda") for v in s])
        y = c.cuda()
        traj = [x.half().cpu()] if want else None
        for idx in range(49, -1, -1):
            t = torch.full((2 * len(x),), int(tmap[idx]), device="cuda")
            out = teacher(torch.cat([x, x]), t, torch.cat([y, torch.full_like(y, 1000)])).float()
            cond, uncond = out[:len(x), :4], out[len(x):, :4]
            eps = torch.cat([uncond[:, :3] + 4 * (cond[:, :3] - uncond[:, :3]), cond[:, 3:4]], 1)
            a, ap = ab[idx].item(), abp[idx].item()
            x0 = (1 / a) ** .5 * x - (1 / a - 1) ** .5 * eps
            eps2 = ((1 / a) ** .5 * x - x0) / (1 / a - 1) ** .5
            x = x0 * ap ** .5 + (1 - ap) ** .5 * eps2
            if want:
                traj.append(x.half().cpu())
        x0s.append(x.cpu())
        if want:
            trajs.append(torch.stack(traj, 1))
        print(f"generated {lo + len(s)}/{n_samples} ({time.time() - start:.0f}s)", flush=True)
    data = dict(x0=torch.cat(x0s), classes=classes, seeds=seeds,
                traj=torch.cat(trajs)[:n_traj], traj_classes=classes[:n_traj])
    torch.save(data, path)
    return data


def smooth_layout(n, gen, device):
    z = torch.randn(n, 1, 4, 4, generator=gen, device=device)
    z = F.avg_pool2d(F.pad(z, (1, 1, 1, 1), mode="replicate"), 3, 1)
    props = torch.distributions.Dirichlet(torch.ones(3)).sample((n,)).to(device)
    q1, q2 = props[:, 0], props[:, 0] + props[:, 1]
    flat = z.flatten(1)
    ranks = flat.argsort(1).argsort(1).float() / 15  # 0..1
    lay = torch.where(ranks < q1[:, None], 8, torch.where(ranks < q2[:, None], 4, 2))
    return lay.reshape(n, 4, 4)


class Batcher:
    def __init__(self, data, teacher, device="cuda"):
        _, create_diffusion = imports()
        self.ab_full = torch.tensor(create_diffusion("").alphas_cumprod, dtype=torch.float32, device=device)
        self.ab50, _, self.tmap = ddim_tables(50, device)
        self.x0 = data["x0"].to(device)
        self.classes = data["classes"].to(device)
        self.traj = data["traj"].to(device)  # N,51,4,32,32 fp16; index j = after j steps
        self.tclasses = data["traj_classes"].to(device)
        self.teacher = teacher
        self.device = device
        self.gen = torch.Generator(device).manual_seed(6220)
        # Calibrate selector threshold range from pooled Eq. 3 scores.
        sub = self.traj[:256].float()
        pooled = []
        for j in range(4, 51):
            d = sub[:, j - 1] - 3 * sub[:, j - 2] + 3 * sub[:, j - 3] - sub[:, j - 4]
            sc = window_scores(d)
            pooled += [sc[4].flatten(), sc[8].flatten()]
        pooled = torch.cat(pooled)
        self.tau_lo, self.tau_hi = pooled.quantile(.02).item(), pooled.quantile(.98).item()

    def tau_sample(self, n):
        u = torch.rand(n, generator=self.gen, device=self.device)
        return math.log(self.tau_lo) + u * (math.log(self.tau_hi) - math.log(self.tau_lo))

    @torch.no_grad()
    def get(self, batch, null_prob=.1, selector_frac=.5):
        g, dev = self.gen, self.device
        half = batch // 2
        # Trajectory states: step i in [0,49] means input latent traj[:, i].
        ti = torch.randint(0, len(self.traj), (half,), generator=g, device=dev)
        si = torch.randint(0, 50, (half,), generator=g, device=dev)
        xa = self.traj[ti, si].float()
        ta = self.tmap.to(dev)[49 - si]
        ya = self.tclasses[ti]
        # Re-noised x0 at uniform t.
        xi = torch.randint(0, len(self.x0), (batch - half,), generator=g, device=dev)
        tb = torch.randint(0, 1000, (batch - half,), generator=g, device=dev)
        a = self.ab_full[tb][:, None, None, None]
        xb = a.sqrt() * self.x0[xi] + (1 - a).sqrt() * torch.randn(self.x0[xi].shape, generator=g, device=dev)
        yb = self.classes[xi]
        x, t, y = torch.cat([xa, xb]), torch.cat([ta, tb]), torch.cat([ya, yb])
        y = torch.where(torch.rand(batch, generator=g, device=dev) < null_prob, 1000, y)
        # Layouts.
        u = torch.rand(batch, generator=g, device=dev)
        layout = smooth_layout(batch, g, dev)
        layout = torch.where((u < .15)[:, None, None], 4, layout)
        layout = torch.where(((u >= .15) & (u < .3))[:, None, None], 8, layout)
        # Selector masks on trajectory states with >=4 latents of history.
        use = (torch.rand(half, generator=g, device=dev) < selector_frac) & (si >= WARMUP)
        if use.any():
            idx = use.nonzero()[:, 0]
            s = si[idx]
            h = [self.traj[ti[idx], (s - k).clamp_min(0)].float() for k in range(4)]
            d = h[0] - 3 * h[1] + 3 * h[2] - h[3]
            sc = window_scores(d)
            taus = self.tau_sample(len(idx)).exp()[:, None, None]
            sel = torch.full_like(sc[4], 2, dtype=torch.long)
            sel = torch.where(sc[4] < taus, 4, sel)
            sel = torch.where(sc[8] < taus, 8, sel)
            layout[idx] = sel
        target = self.teacher(x, t, y).float()
        return x, t, y, layout, target


def loss_fn(pred, target):
    return F.mse_loss(pred[:, :4], target[:, :4]) + .1 * F.mse_loss(pred[:, 4:], target[:, 4:])


@torch.no_grad()
def validate(model, val):
    model.eval()
    out = {}
    for name, lay in val["layouts"].items():
        losses = []
        for lo in range(0, len(val["x"]), 256):
            sl = slice(lo, lo + 256)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(val["x"][sl], val["t"][sl], val["y"][sl], lay[sl])
            losses.append(F.mse_loss(pred.float()[:, :4], val["target"][sl][:, :4]).item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def preview(model, teacher, folder, step, taus, lpips_fn, vae):
    """16 fresh-seed images: teacher vs uniform4/8 and joint at mid tau."""
    model.eval()
    seeds = list(range(90_000, 90_016))
    classes = [207, 360, 387, 974, 88, 979, 417, 279, 1, 9, 33, 130, 250, 291, 323, 980]
    rows, res = [], {}
    ref, _ = sample_batch(model, seeds, classes, arm="fine", base=teacher)
    ref_img = vae.decode(ref / .18215).sample.clamp(-1, 1)
    rows.append(ref_img)
    for arm, tau in (("uniform4", None), ("uniform8", None), ("joint", taus[1])):
        lat, st = sample_batch(model, seeds, classes, arm=arm, tau=tau, dtype=torch.bfloat16)
        img = vae.decode(lat / .18215).sample.clamp(-1, 1)
        rows.append(img)
        res[arm] = dict(lpips=lpips_fn(img, ref_img).mean().item(), tokens=st["tokens"].mean().item())
    save_image(torch.cat(rows), folder / f"preview_{step:06d}.png", nrow=16, normalize=True, value_range=(-1, 1))
    model.train()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="mixed_lora32")
    p.add_argument("--samples", type=int, default=50_000)
    p.add_argument("--traj", type=int, default=5_000)
    p.add_argument("--minutes", type=float, default=75)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--full-finetune", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--preview-every", type=int, default=2000)
    p.add_argument("--gen-only", action="store_true")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(6220)
    teacher = load_dit(CKPT, "cuda")
    data = generate(teacher, args.samples, args.traj, batch=1000)
    if args.gen_only:
        return
    folder = ROOT / "artifacts/training" / args.name
    folder.mkdir(parents=True, exist_ok=True)
    batcher = Batcher(data, teacher)
    model = MixedPatchDiT(load_dit(CKPT), rank=args.rank, full_finetune=args.full_finetune).cuda()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu", weights_only=True)["model"])
    params = model.trainable_parameters()
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0., betas=(.9, .99))
    # Fixed validation set: held-out trajectories are the last 256 (excluded from training draws below).
    vg = torch.Generator("cuda").manual_seed(1)
    nval = 1024
    vt = torch.randint(len(batcher.traj) - 256, len(batcher.traj), (nval,), generator=vg, device="cuda")
    vs = torch.randint(0, 50, (nval,), generator=vg, device="cuda")
    val = dict(x=batcher.traj[vt, vs].float(), t=batcher.tmap.cuda()[49 - vs], y=batcher.tclasses[vt])
    val["target"] = torch.cat([teacher(val["x"][i:i + 256], val["t"][i:i + 256], val["y"][i:i + 256]) for i in range(0, nval, 256)]).float()
    val["layouts"] = dict(uniform4=torch.full((nval, 4, 4), 4, device="cuda"),
                          uniform8=torch.full((nval, 4, 4), 8, device="cuda"),
                          mixed=smooth_layout(nval, vg, "cuda"))
    batcher.traj = batcher.traj[:-256]
    batcher.tclasses = batcher.tclasses[:-256]
    import lpips
    from .evaluate import load_vae
    lpips_fn = lpips.LPIPS(net="alex").cuda().eval()
    vae = load_vae().cuda()
    taus = [math.exp(v) for v in torch.linspace(math.log(batcher.tau_lo), math.log(batcher.tau_hi), 3).tolist()]
    config = vars(args) | dict(trainable_parameters=n_params, tau_range=[batcher.tau_lo, batcher.tau_hi],
                               train_x0=len(batcher.x0), train_traj=len(batcher.traj), val_states=nval,
                               objective="MSE(eps, 4ch) + 0.1 MSE(var), teacher targets, bf16 autocast")
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    log = []
    start = time.time()
    deadline = start + args.minutes * 60
    step = 0
    warm = 500
    est_steps = None
    model.train()
    while time.time() < deadline:
        x, t, y, layout, target = batcher.get(args.batch)
        frac = (time.time() - start) / (args.minutes * 60)
        lr = args.lr * min(1, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(frac, 1))) * .95 + .05)
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x, t, y, layout)
        loss = loss_fn(pred.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"step={step} loss={loss.item():.5f} lr={lr:.2e} elapsed={time.time() - start:.0f}s", flush=True)
        if step % 500 == 0:
            v = validate(model, val)
            log.append(dict(step=step, seconds=time.time() - start, loss=loss.item(), val=v))
            print(f"val step={step}: {v}", flush=True)
        if step % args.preview_every == 0:
            r = preview(model, teacher, folder, step, taus, lpips_fn, vae)
            log.append(dict(step=step, seconds=time.time() - start, preview=r))
            print(f"preview step={step}: {r}", flush=True)
            torch.save(dict(model=model.state_dict(), rank=args.rank, full_finetune=args.full_finetune,
                            step=step, tau_range=[batcher.tau_lo, batcher.tau_hi]), folder / "last.pt")
            (folder / "log.json").write_text(json.dumps(log, indent=2))
    v = validate(model, val)
    r = preview(model, teacher, folder, step, taus, lpips_fn, vae)
    log.append(dict(step=step, seconds=time.time() - start, val=v, preview=r))
    torch.save(dict(model=model.state_dict(), rank=args.rank, full_finetune=args.full_finetune,
                    step=step, tau_range=[batcher.tau_lo, batcher.tau_hi]), folder / "last.pt")
    (folder / "log.json").write_text(json.dumps(log, indent=2))
    print(f"done step={step} val={v} preview={r}", flush=True)


if __name__ == "__main__":
    main()
