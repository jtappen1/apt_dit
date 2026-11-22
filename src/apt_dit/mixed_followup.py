"""Follow-up B (see results/mixed/preregistration_followup.md).

  damage   per-image single-block p=8 damage maps + causal step-3 features
  oracle   B1 stop/go: placement by measured damage vs random placement
  train    B2: small CNN predicting the damage map from causal features
  learned  B2: allocation gap + end-to-end placement by the learned predictor
"""
import argparse
import json
import math

import numpy as np
import torch
from scipy import stats as sstats
from torch import nn
from torch.nn import functional as F

from .mixed_eval import (OUT, boot_ci, eval_set, forced_block, jdump, load_models, tau_for,
                         teacher_signals, block_variation)
from .mixed_sample import WARMUP, ddim_tables, initial_noise, sample_batch, window_scores

FOLD = OUT / "followup"
SPLITS = dict(train=(3000, 900_000), eval=(2000, 950_000))


@torch.no_grad()
def warmup_features(teacher, seeds, classes, cfg=4.):
    """Causal inputs available at the first scheduled step: x_t, x0-hat, Delta^3 (12x32x32)."""
    x = initial_noise(seeds, "cuda")
    y = torch.as_tensor(classes, device="cuda")
    B = len(x)
    ab, abp, tmap = ddim_tables(50, "cuda")
    hist = [x]
    for i, idx in enumerate(range(49, 49 - WARMUP - 1, -1)):
        t = torch.full((2 * B,), int(tmap[idx]), device="cuda")
        out = teacher(torch.cat([x, x]), t, torch.cat([y, torch.full_like(y, 1000)])).float()
        cond, uncond = out[:B, :4], out[B:, :4]
        eps = torch.cat([uncond[:, :3] + cfg * (cond[:, :3] - uncond[:, :3]), cond[:, 3:4]], 1)
        a, ap = ab[idx].item(), abp[idx].item()
        x0 = (1 / a) ** .5 * x - (1 / a - 1) ** .5 * eps
        if i == WARMUP:  # prediction at the first scheduled step; do not advance
            break
        eps2 = ((1 / a) ** .5 * x - x0) / (1 / a - 1) ** .5
        x = x0 * ap ** .5 + (1 - ap) ** .5 * eps2
        hist.append(x)
    d = hist[-1] - 3 * hist[-2] + 3 * hist[-3] - hist[-4]
    return torch.cat([hist[-1], x0, d], 1), window_scores(d)[8]


@torch.no_grad()
def decode(vae, lat, chunk=50):
    return torch.cat([vae.decode(lat[j:j + chunk] / .18215).sample.clamp(-1, 1) for j in range(0, len(lat), chunk)])


@torch.no_grad()
def damage(args):
    import lpips
    from .evaluate import load_vae
    torch.backends.cuda.matmul.allow_tf32 = True
    model, teacher, _ = load_models(args.checkpoint)
    vae = load_vae().cuda()
    lp = lpips.LPIPS(net="alex").cuda().eval()
    for split in args.splits:
        path = FOLD / f"damage_{split}.pt"
        if path.exists():
            continue
        n, off = SPLITS[split]
        seeds, classes = eval_set(n, offset=off)
        D, feats, fd8_causal, sig, refs = [], [], [], {"fd8_mean": [], "variation_mean": []}, []
        for lo in range(0, n, args.batch):
            s, c = seeds[lo:lo + args.batch], classes[lo:lo + args.batch]
            ref_lat, sg = teacher_signals(teacher, s, c)
            ref = decode(vae, ref_lat)
            if split == "eval":
                refs.append(ref.half().cpu())
            f, fd = warmup_features(teacher, s, c)
            feats.append(f.half().cpu())
            fd8_causal.append(fd.flatten(1).cpu())
            sig["fd8_mean"].append(sg["fd8"].flatten(1).cpu())
            sig["variation_mean"].append(sg["variation"].flatten(1).cpu())
            dmg = torch.zeros(len(s), 16)
            for blk in range(16):
                lat = forced_block(model, s, c, blk, 8)
                img = decode(vae, lat)
                dmg[:, blk] = torch.cat([lp(img[j:j + 100], ref[j:j + 100]).flatten() for j in range(0, len(img), 100)]).cpu()
            D.append(dmg)
            print(f"damage {split}: {lo + len(s)}/{n}", flush=True)
        out = dict(seeds=seeds, classes=classes, damage=torch.cat(D), feats=torch.cat(feats),
                   fd8_causal=torch.cat(fd8_causal), **{k: torch.cat(v) for k, v in sig.items()})
        if refs:
            out["ref_imgs"] = torch.cat(refs)
        FOLD.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)


def paired(a, b, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
    return dict(mean=float(d.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))])


def gap_of(S, D):
    rhos = np.nan_to_num(np.array([sstats.spearmanr(S[i], D[i]).statistic for i in range(len(D))]))
    lo, hi = boot_ci(rhos)
    return dict(spearman=float(rhos.mean()), spearman_ci=[lo, hi], gap=float((1 - rhos.mean()) / 2),
                gap_ci=[(1 - hi) / 2, (1 - lo) / 2])


@torch.no_grad()
def run_arms(model, vae, lp, data, specs, batch):
    seeds, classes, refs = data["seeds"], data["classes"], data["ref_imgs"]
    res = {}
    for name, kw in specs:
        vals, toks = [], []
        for lo in range(0, len(seeds), batch):
            sl = slice(lo, lo + batch)
            kw2 = dict(kw)
            if "priority" in kw2:
                kw2["priority"] = kw2["priority"][sl].cuda()
            lat, st = sample_batch(model, seeds[sl], classes[sl], **kw2)
            img = torch.cat([vae.decode(lat[j:j + 100] / .18215).sample.clamp(-1, 1) for j in range(0, len(lat), 100)])
            vals.append(lp(img, refs[sl].cuda().float()).flatten().cpu())
            toks.append(st["tokens"])
        v = torch.cat(vals).numpy()
        res[name] = dict(lpips=float(v.mean()), lpips_ci=list(boot_ci(v)),
                         mean_tokens=float(torch.cat(toks).float().mean()), per_image=v.tolist())
        print(f"{name}: LPIPS={v.mean():.4f} tokens={res[name]['mean_tokens']:.1f}", flush=True)
    return res


def arm_specs(cal, priority, tag):
    specs = []
    for T in (100, 70):
        tau = tau_for(cal["joint"], T)
        specs += [(f"joint_T{T}", dict(arm="joint", tau=tau)),
                  (f"joint_random_T{T}", dict(arm="joint_random", tau=tau)),
                  (f"joint_{tag}_T{T}", dict(arm="joint_prio", tau=tau, priority=priority)),
                  (f"spatial_random_T{T}", dict(arm="spatial_random", target=T)),
                  (f"spatial_{tag}_T{T}", dict(arm="spatial_prio", target=T, priority=priority))]
    return specs


def contrasts(res, tag):
    out = {}
    for T in (100, 70):
        out[f"joint_{tag}-joint_random_T{T}"] = paired(res[f"joint_{tag}_T{T}"]["per_image"], res[f"joint_random_T{T}"]["per_image"])
        out[f"spatial_{tag}-spatial_random_T{T}"] = paired(res[f"spatial_{tag}_T{T}"]["per_image"], res[f"spatial_random_T{T}"]["per_image"])
        out[f"joint_{tag}-joint_T{T}"] = paired(res[f"joint_{tag}_T{T}"]["per_image"], res[f"joint_T{T}"]["per_image"])
        out[f"joint-joint_random_T{T}"] = paired(res[f"joint_T{T}"]["per_image"], res[f"joint_random_T{T}"]["per_image"])
    return out


def common(args):
    import lpips
    from .evaluate import load_vae
    torch.backends.cuda.matmul.allow_tf32 = True
    model, teacher, _ = load_models(args.checkpoint)
    vae = load_vae().cuda()
    lp = lpips.LPIPS(net="alex").cuda().eval()
    cal = json.loads((OUT / args.cal_tag / "calibration.json").read_text())
    data = torch.load(FOLD / "damage_eval.pt", map_location="cpu", weights_only=False)
    return model, vae, lp, cal, data


@torch.no_grad()
def oracle(args):
    model, vae, lp, cal, data = common(args)
    pri = data["damage"].reshape(-1, 4, 4).float()
    res = run_arms(model, vae, lp, data, arm_specs(cal, pri, "oracle"), args.batch)
    con = contrasts(res, "oracle")
    go = any(con[f"{k}_oracle-{k}_random_T{T}"]["ci"][1] < -0.01 for k in ("joint", "spatial") for T in (100, 70))
    summary = dict(contrasts=con, go=go, arms={k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in res.items()})
    jdump(FOLD / "oracle.json", summary)
    torch.save(res, FOLD / "oracle_per_image.pt")
    print(json.dumps(dict(contrasts=con, go=go), indent=2), flush=True)


class DamageNet(nn.Module):
    def __init__(self, yemb):
        super().__init__()
        self.register_buffer("yemb", yemb)  # frozen DiT class table [1001, 384]
        self.y = nn.Linear(yemb.shape[1], 128)
        self.net = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),   # 16
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(),  # 8
            nn.Conv2d(128, 128, 3, stride=2, padding=1))            # 4
        self.head = nn.Sequential(nn.SiLU(), nn.Conv2d(128, 64, 1), nn.SiLU(), nn.Conv2d(64, 1, 1))

    def forward(self, f, y):
        h = self.net(f) + self.y(self.yemb[y])[:, :, None, None]
        return self.head(h)[:, 0]


def zscore(d):
    return (d - d.mean(1, keepdim=True)) / (d.std(1, keepdim=True) + 1e-6)


def train(args):
    from .upstream import load_dit
    from .mixed_eval import CKPT
    tr = torch.load(FOLD / "damage_train.pt", map_location="cpu", weights_only=False)
    yemb = load_dit(CKPT).y_embedder.embedding_table.weight.detach().clone()
    f = tr["feats"].float().cuda()
    f = f / f.flatten(2).std(-1).mean(0)[None, :, None, None]  # per-channel scale
    scale = tr["feats"].float().flatten(2).std(-1).mean(0)
    y = torch.as_tensor(tr["classes"]).cuda()
    D = zscore(tr["damage"].float()).reshape(-1, 4, 4).cuda()
    n = len(f)
    va = torch.arange(n - 500, n, device="cuda")
    trn = torch.arange(0, n - 500, device="cuda")
    torch.manual_seed(6220)
    net = DamageNet(yemb).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state, log = -1, None, []

    def val_rho():
        net.eval()
        with torch.no_grad():
            p = net(f[va], y[va]).flatten(1).cpu().numpy()
        net.train()
        return gap_of(p, D[va].flatten(1).cpu().numpy())["spearman"]

    for step in range(1, args.steps + 1):
        idx = trn[torch.randint(0, len(trn), (128,), device="cuda")]
        xb, db = f[idx], D[idx]
        flip = torch.rand(len(idx), device="cuda") < .5
        xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
        db = torch.where(flip[:, None, None], db.flip(-1), db)
        loss = F.mse_loss(net(xb, y[idx]), db)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            r = val_rho()
            log.append(dict(step=step, loss=loss.item(), val_spearman=r))
            if r > best:
                best, best_state = r, {k: v.clone() for k, v in net.state_dict().items()}
            print(f"step {step} loss {loss.item():.4f} val_rho {r:.3f}", flush=True)
    torch.save(dict(model=best_state, scale=scale, val_spearman=best), FOLD / "damagenet.pt")
    jdump(FOLD / "damagenet_log.json", log)


@torch.no_grad()
def learned(args):
    from .upstream import load_dit
    from .mixed_eval import CKPT
    model, vae, lp, cal, data = common(args)
    ck = torch.load(FOLD / "damagenet.pt", map_location="cpu", weights_only=False)
    net = DamageNet(load_dit(CKPT).y_embedder.embedding_table.weight.detach().clone()).cuda().eval()
    net.load_state_dict(ck["model"])
    f = data["feats"].float() / ck["scale"][None, :, None, None]
    y = torch.as_tensor(data["classes"])
    pred = torch.cat([net(f[i:i + 500].cuda(), y[i:i + 500].cuda()).cpu() for i in range(0, len(f), 500)])
    D = data["damage"].float().numpy()
    gaps = dict(learned=gap_of(pred.flatten(1).numpy(), D),
                fd8_causal=gap_of(data["fd8_causal"].numpy(), D),
                fd8_mean=gap_of(data["fd8_mean"].numpy(), D),
                variation_mean=gap_of(data["variation_mean"].numpy(), D),
                random=gap_of(np.random.default_rng(0).random(D.shape), D),
                oracle=gap_of(D, D))
    print(json.dumps(gaps, indent=2), flush=True)
    res = run_arms(model, vae, lp, data, [s for s in arm_specs(cal, pred, "learned") if "learned" in s[0]], args.batch)
    prev = torch.load(FOLD / "oracle_per_image.pt", weights_only=False)
    for k in ("joint_T100", "joint_random_T100", "spatial_random_T100", "joint_T70", "joint_random_T70", "spatial_random_T70"):
        res[k] = prev[k]
    con = contrasts(res, "learned")
    jdump(FOLD / "learned.json", dict(gaps=gaps, contrasts=con,
                                      arms={k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in res.items()}))
    print(json.dumps(con, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["damage", "oracle", "train", "learned"])
    p.add_argument("--checkpoint", default=str(OUT.parents[0] / "training/mixed_lora32/last.pt"))
    p.add_argument("--cal-tag", default="aggr")
    p.add_argument("--splits", nargs="+", default=["eval", "train"])
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--steps", type=int, default=3000)
    args = p.parse_args()
    dict(damage=damage, oracle=oracle, train=train, learned=learned)[args.cmd](args)


if __name__ == "__main__":
    main()
