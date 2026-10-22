"""Driver for batched runs of bfl.run_batch: large batches (different methods/shares/seeds in one batch), a GPU guard (waits while the GPU is busy with someone else's job), appends to jsonl.
Usage: python driver.py --dataset pems_bay --attack hide --methods fedavg,probe2 --shares 0.2,0.4,0.6 --seeds 0,1,2,3,4 --norm 1 --delta 0.5 --tau 0.25 --chunk 40 --out ../results/final_pems_bay.jsonl [--clean 1]"""
import argparse, json, time, sys
import torch
import bfl

GPU = "/sys/class/drm/card1/device/gpu_busy_percent"


def gpu_guard(limit=30, max_wait=6 * 3600):
    """Wait until the GPU is free (foreign load > limit %). Our runs go between calls, so we measure right before the start of a batch."""
    import os
    if os.environ.get("PT_NOGUARD"): return
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(2)      # after our batch the counter still holds its load
        try: v = [int(open(GPU).read()) for _ in range(5) if not time.sleep(1)]
        except OSError: return
        if sorted(v)[2] <= limit: return          # median
        print(f"  GPU busy ({v} %), waiting 60 s", flush=True); time.sleep(60)


def run(ctx, specs, attack, out, chunk=40, **kw):
    for i in range(0, len(specs), chunk):
        gpu_guard(); ch = specs[i:i + chunk]; ni = kw.pop('noniid', False); res, t = bfl.run_batch(ctx, ch, 25, 20, 64, 0.003, attack, bool(ni), None, False, **kw); kw['noniid'] = ni
        with open(out, "a") as fh:
            for o in res: o["time_batch_s"] = round(t, 1); fh.write(json.dumps(o) + "\n")
        print(f"{ctx.dataset} {attack} batch of {len(ch)}: {t:.0f} s ({t/len(ch):.1f} s/run)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--attack", default="hide"); ap.add_argument("--methods", required=True); ap.add_argument("--shares", default="0.2,0.4,0.6")
    ap.add_argument("--seeds", default="0,1,2,3,4"); ap.add_argument("--clean", type=int, default=0); ap.add_argument("--norm", type=int, default=1); ap.add_argument("--delta", type=float, default=0.5); ap.add_argument("--tau", type=float, default=0.25)
    ap.add_argument("--hide_mag", type=float, default=25.0); ap.add_argument("--chunk", type=int, default=40); ap.add_argument("--adapt_c", type=float, default=0.5); ap.add_argument("--collude", type=int, default=0); ap.add_argument("--lam", type=float, default=5.0); ap.add_argument("--aware_k", type=int, default=24)
    ap.add_argument("--root_n", type=int, default=10); ap.add_argument("--root_mode", default="random"); ap.add_argument("--root_poison", type=int, default=0); ap.add_argument("--noniid", type=int, default=0); ap.add_argument("--out", required=True); a = ap.parse_args()
    dev = torch.device("cuda"); ctx = bfl.Ctx(dev, a.dataset); S = [int(x) for x in a.seeds.split(",")]; M = a.methods.split(",")
    specs = [dict(method=m, condition="clean", seed=s, bad=0.2) for m in M for s in S] if a.clean else [dict(method=m, condition="atk", seed=s, bad=float(b)) for b in a.shares.split(",") for m in M for s in S]
    run(ctx, specs, a.attack if not a.clean else "hide", a.out, a.chunk, delta=a.delta, tau=a.tau, hide_mag=a.hide_mag, norm=bool(a.norm), adapt_c=a.adapt_c, collude=bool(a.collude), lam=a.lam, aware_k=a.aware_k, root_n=a.root_n, root_mode=a.root_mode, root_poison=a.root_poison, noniid=a.noniid)
