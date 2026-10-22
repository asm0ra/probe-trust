"""Choice of normalised thresholds (delta, tau as fractions of the network speed sigma) ONLY on METR-LA, seeds 100-101, attacks hide and exagg (share 0.4) + clean runs; other networks and attacks serve as a check without tuning."""
import json, itertools, torch, bfl
from driver import run
ctx = bfl.Ctx(torch.device("cuda"), "metr_la"); out = "../results/norm_tune.jsonl"
for dk, tk in itertools.product([0.25, 0.5, 0.75, 1.0], [0.1, 0.25, 0.5, 1.0]):
    for atk in ("hide", "exagg"):
        specs = [dict(method=m, condition=c, seed=s, bad=0.4) for m in ("probe2",) for c in ("atk", "clean") for s in (100, 101)]
        run(ctx, specs, atk, out, 8, delta=dk, tau=tk, norm=True)
for atk in ("hide", "exagg"):      # reference values
    run(ctx, [dict(method=m, condition=c, seed=s, bad=0.4) for m in ("fedavg", "fltrust") for c in ("atk", "clean") for s in (100, 101)], atk, out, 8, delta=0.5, tau=0.25, norm=True)
